import argparse
from datetime import datetime
import math
import os
import random
from time import sleep, time
import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from airsim.types import AngleLevelControllerGains, PIDGains, AngleRateControllerGains
import numpy as np
from tqdm import tqdm
import cv2
cv2.setNumThreads(1)
cv2.namedWindow("erp_depth", cv2.WINDOW_NORMAL)
try: cv2.startWindowThread()
except: pass

import msgpackrpc
import torch
import torch.nn.functional as F
from panoramic_utils import simple_cubemap_to_erp_rgb
from model import Model

from unik3d_utlis import depth_infer
import matplotlib
matplotlib.use("TkAgg")   
from unik3d.models import UniK3D

# ==== Tools ====
def connect_manager(ip: str, port: int, scen_ids):
   
    cli = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout=180)
    assert cli.call("ping") is True
    ok, ret = cli.call("reopen_scenes", ip, list(scen_ids))
    assert ok, "reopen_scenes failed"
    airsim_ip, ports = ret
    if isinstance(airsim_ip, bytes):
        airsim_ip = airsim_ip.decode()
    return airsim_ip, [int(p) for p in ports]


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    quaternions: (..., 4) with real part first
    return: (..., 3, 3)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



# ==== Parameters ====
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='/home/insta360/Fly360/weights/fly360.pth')
parser.add_argument('--env', default='Forest')
parser.add_argument('--max_speed', default=10.0, type=float, help='(m/s) real speed might be xx m/s slower')
parser.add_argument('--margin', default=0.35, type=float, help='(m) radius of body')
parser.add_argument('--env_scene', default=1, type=int)
parser.add_argument('--port', default=30000, type=int)


args = parser.parse_args()
hover_thr = 0.65
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'exps/{datetime_str}/'
os.makedirs(log_dir)

# Here we can configure multiple drones

def circle_traj(cx, cy, r, z=-4.0, step_deg=10, start_deg=90, clockwise=False, close=False):
    """
    Generate circular trajectory points around (cx, cy) with radius r and fixed z, step_deg degrees per step.
    - start_deg: starting angle (degrees), 0° corresponds to (cx+r, cy)
    - clockwise: whether to rotate clockwise (default False=counter-clockwise)
    - close: whether to add the starting point again to the end to close the circle
    Return: [[x, y, z], ...], each value rounded to 3 decimal places (converted to float, displayed as x.x)
    """
    sgn = -1 if clockwise else 1
    pts = []
    for i in range(0, 360, step_deg):
        a = math.radians(start_deg + sgn * i)
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        pts.append([float(f"{x:.3f}"), float(f"{y:.3f}"), float(f"{z:.3f}")])
    if close:
        pts.append(pts[0])
    return pts

cir1 =  circle_traj(20, 0, 18.0, z=-3, step_deg=5, start_deg=90, clockwise=False, close=False)
agents = {
    "Forest": [
        ("Drone_1", [[-800, -10, -7], [-900, -10, -7],[-800, -10, -7], [-900, -10, -7],[-800, -10, -7], [-900, -10, -7],[-800, -10, -7]]),  
    ],
    "Park": [
        ("Drone_1", [[-5, 0, -2.5], [35, 0, -2.5],[-5, 0, -2.5], [35, 0, -2.5],[-5, 0, -2.5], [35, 0, -2.5],[-5, 0, -2.5]]) ], # or cir1
}[args.env]

B = len(agents)
agent_names, agent_waypoints = zip(*agents)
target_pos = [w[-1] for w in agent_waypoints]
traj_history = {agent_name: [] for agent_name in agent_names}


# ==== 连接 AirSim ====
airsim_ip, ports = connect_manager("127.0.0.1", 30000, [args.env_scene])
print(f"Connected to manager service, IP: {airsim_ip}, port: {ports}")
airsim_port = ports[0]
print(f"Using AirSim port: {airsim_port}")

client = airsim.MultirotorClient(ip=airsim_ip, port=airsim_port)
client.confirmConnection()
print("AirSim connected successfully!")
client.reset()
print("Vehicles:", client.listVehicles())


# ==== Model ====
device = torch.device('cuda')
model = Model(10, 6,input_channels=1).eval().to(device)
if args.resume:
    # PyTorch safety hint: future consideration of weights_only=True
    model.load_state_dict(torch.load(args.resume, map_location=device))


type_ = "s"  # available types: s, b, l
name = f"unik3d-vit{type_}"
depth_model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
depth_model.resolution_level = 9
depth_model.interpolation_mode = "bilinear"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model = depth_model.to(device).eval()

@torch.no_grad()
def main():
    h = None
    for _ in range(10):
        _, _, h = model(
            torch.zeros(B, 1, 64, 128, device=device),
            torch.zeros(B, model.v_proj.in_features, device=device),
            h)

    sleep(1)
    # Initialize drones
    # We can also set multiple drones here
    for agent_name, waypoints in agents:
        sleep(0.1)
        # print(agent_name)
        # import pdb; pdb.set_trace()
        client.enableApiControl(True, agent_name)
        client.armDisarm(True, agent_name)

        client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=agent_name)
        yaw = math.atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
        start_pt = waypoints.pop(0)
        start_pt = [
            start_pt[0] + random.random() * 0.2 - 0.1,
            start_pt[1] + random.random() * 0.2 - 0.1,
            start_pt[2] + random.random() * 0.5 - 0.25,
        ]
        for _ in range(3):
            sleep(0.1)
            client.simSetVehiclePose(Pose(
                Vector3r(*start_pt),
                Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))),
                ignore_collision=True, vehicle_name=agent_name)
        client.setAngleRateControllerGains(AngleRateControllerGains(
            roll_gains=PIDGains(0.2, 0.01, 0.001),
            pitch_gains=PIDGains(0.2, 0.01, 0.001),
            yaw_gains=PIDGains(0.25, 0.00, 0.001),
        ))
        client.setAngleLevelControllerGains(AngleLevelControllerGains(
            roll_gains=PIDGains(2, 0, 0),
            pitch_gains=PIDGains(2, 0, 0),
            yaw_gains=PIDGains(2, 0, 0),
        ), agent_name)

        

    sleep(1)
    for agent_name, _ in agents:
        client.simGetCollisionInfo(agent_name)

    p_target = torch.empty((B, 3))
    last_p = torch.empty((B, 3))
    forward_vec = torch.empty((B, 3))
    v = torch.empty((B, 3))
    R = torch.empty((B, 3, 3))
    traveled_distance = [0 for _ in agents]
    traveled_time = [0 for _ in agents]
    done_flag = [False for _ in agents]
    has_collided = [set() for _ in agents]
    extra = torch.tensor([[args.margin]]).repeat(B, 1)

    for i, (agent_name, waypoints) in enumerate(agents):
        x, y, z = waypoints.pop(0)
        # Unified coordinate system conversion: consistent with runtime
        p_target[i] = torch.as_tensor([x, -y, -z])

        state = client.getMultirotorState(agent_name)
        q = state.kinematics_estimated.orientation
        p = state.kinematics_estimated.position

        q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
        last_p[i] = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
        forward_vec[i] = quaternion_to_matrix(q)[:, 0]

    pbar = tqdm()
    hidden_state = None
    t_begin_real = time()
    state = client.getMultirotorState(agent_names[0])
    t_now = t_begin = state.timestamp / 1e9
    t_end = t_begin + 12000
    ctl_error = 0
    ctl_error = torch.randn((len(agents), 3)) * 0.17

    while t_now < t_end:
        pbar.update()
        cam_names = ["cube_front", "cube_right", "cube_left", "cube_back", "cube_up", "cube_down"]

        # Request Scene type non-float, non-compressed raw byte data
        req6 = [airsim.ImageRequest(c, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                for c in cam_names]

        rgb_cubemaps = []  # [(6,H,W,3)] for each agent
        for agent in agent_names:
            # One RPC, return 6 images
            responses = client.simGetImages(req6, vehicle_name=agent)
            faces = []
            for resp in responses:
                h, w = resp.height, resp.width
                if h <= 0 or w <= 0 or len(resp.image_data_uint8) == 0:
                    faces.append(np.zeros((h if h>0 else 1, w if w>0 else 1, 3), dtype=np.uint8))
                    continue

                # Convert continuous bytes to ndarray
                buf = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                # Check channel number (AirSim commonly uses BGR(3) or BGRA(4))
                expected3 = h * w * 3
                expected4 = h * w * 4
                if buf.size == expected3:
                    img = buf.reshape(h, w, 3)              # BGR
                elif buf.size == expected4:
                    img = buf.reshape(h, w, 4)[:, :, :3]    # BGRA -> BGR
                else:
                    # Size mismatch, give a placeholder image
                    faces.append(np.zeros((h, w, 3), dtype=np.uint8))
                    continue

                # Convert to RGB
                img_rgb = img[:, :, ::-1].copy()  # BGR -> RGB
                faces.append(img_rgb.astype(np.uint8))

            # (6, H, W, 3)
            rgb_cubemaps.append(np.stack(faces, axis=0))

        # (B, 6, H, W, 3)
        rgb_np = np.stack(rgb_cubemaps, axis=0)
        rgb_tensor = torch.from_numpy(rgb_np[0]).float().to(device) 
        erp_rgb = simple_cubemap_to_erp_rgb(rgb_tensor) # 1,3,224,448
        erp_depth = depth_infer(depth_model, erp_rgb[0], [448.0, 224.0, 3.14159, 3.14159/2.0]) # 1,1,224,448


        # Convert to numpy for display
        erp_img = erp_rgb[0].permute(1, 2, 0).cpu().numpy()/ 255.0     # H,W,3 (RGB)
        erp_bgr = erp_img[..., ::-1]                          # RGB->BGR

        # ===== Visualize depth =====
        depth_np = erp_depth.squeeze().detach().cpu().numpy().astype(np.float32)  # (H,W)
        depth_np = np.clip(depth_np, None, 100)
        dmin, dmax = depth_np.min(), depth_np.max()
        depth_norm = (depth_np - dmin) / (dmax - dmin + 1e-8)                      # [0,1]
        depth_u8 = (depth_norm * 255).astype(np.uint8)                             # [0,255] uint8
        depth_color = cv2.applyColorMap(depth_u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))


        # # # Display
        cv2.imshow("ERP_RGB", erp_bgr)
        cv2.imshow("erp_depth", depth_color)
        cv2.waitKey(1)

        # ---- Control logic  ----
        for i, (agent_name, waypoints) in enumerate(agents):
            state = client.getMultirotorState(agent_name)
            t_now = state.timestamp / 1e9
            p = state.kinematics_estimated.position
            q = state.kinematics_estimated.orientation
            _v = state.kinematics_estimated.linear_velocity
            traj_history[agent_name].append([p.x_val, -p.y_val, -p.z_val, q.w_val, q.x_val, -q.y_val, -q.z_val])
            

            p = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
            duration = t_now - t_begin
            if not done_flag[i]:
                traveled_distance[i] += torch.norm(p - last_p[i]).item()
                traveled_time[i] = duration
            last_p[i] = p
            v[i] = torch.as_tensor([_v.x_val, -_v.y_val, -_v.z_val])

            q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
            R[i] = quaternion_to_matrix(q)

            if not done_flag[i] and torch.norm(p_target[i] - p) < 5:
                if waypoints:
                    x, y, z = waypoints.pop(0)
                    p_target[i] = torch.as_tensor([x, -y, -z])

        # target_v = p_target - last_p
        target_v = (p_target - last_p)
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v = target_v / target_v_norm * target_v_norm.clamp_max(args.max_speed)
        
        env_R = R.clone()  # (B,3,3)

        # ====== State input calculation (use env_R, unified dimension) ======
        tv_world = torch.squeeze(target_v[:, None] @ env_R, 1)   # (B, 3)
        up_world = env_R[:, :, 2]                                # (B, 3)
        extra = extra.view(B, 1)                                 # (B, 1) 
        local_v = torch.squeeze(v[:, None] @ env_R, 1)           # (B, 3)

        state_vecs = [tv_world, up_world, extra]
        state_vecs.insert(0, local_v)

        state_cat = torch.cat(state_vecs, dim=-1).to(device)    

        depth_t = torch.from_numpy(depth_norm).float().to(device)   
        depth_t = depth_t.unsqueeze(0)
        d_min, d_max = 0.01, 30.0
        depth_t = depth_t * (d_max - d_min) + d_min
        depth_t = 3.0 / depth_t.clamp_(0.3, 24.0) - 0.6
        depth_t = depth_t.unsqueeze(0)
        depth_t = F.interpolate(depth_t, size=(64, 128), mode='bilinear', align_corners=False)  

        x = depth_t

        state_cat = state_cat.to(device)
        action, _, hidden_state = model(x, state_cat, hidden_state)
        v_setpoint, v_est = (env_R @ action.cpu().reshape(B, 3, -1)).unbind(-1)
        
        action_body = action.cpu().reshape(B, 3, -1)
        v_setpoint_body, v_est_body = action_body.unbind(-1)
        a_setpoint = v_setpoint - v_est + ctl_error
        a_setpoint[:, 2] += 9.80665  

        # Thrust magnitude
        throttle = torch.norm(a_setpoint, 2, -1)
        up_vec = a_setpoint / throttle[..., None]
        throttle = throttle + local_v[:, 2] * local_v[:, 2].abs() * 0.01

        # === Fixed yaw, only use up_vec (from expected acceleration) to solve roll/pitch ===
        FIXED_YAW_RAD =math.pi*(1)  # e.g. 0 degree;
        yaw = torch.full((B,), FIXED_YAW_RAD, device=up_vec.device) 
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        f0 = torch.stack([cy, sy, torch.zeros_like(cy)], dim=-1)      # forward0
        l0 = torch.stack([-sy, cy, torch.zeros_like(cy)], dim=-1)     # left
        u_f = (up_vec * f0).sum(-1)           
        u_l = (up_vec * l0).sum(-1)          
        u_l = u_l.clamp(-1 + 1e-6, 1 - 1e-6)   
        roll = torch.asin(-u_l)
        cr = torch.sqrt(torch.clamp(1 - torch.sin(roll) ** 2, min=1e-6))
        sp = (u_f / cr).clamp(-1 + 1e-6, 1 - 1e-6)
        pitch = torch.asin(sp)

        print(f"roll: {roll.cpu().numpy()}, pitch: {pitch.cpu().numpy()}, yaw: {yaw.cpu().numpy()}")

        def normalize_angle(angle):
            return (angle + math.pi) % (2 * math.pi) - math.pi

        for i, (r, p, y, t) in enumerate(zip(roll.tolist(), pitch.tolist(), yaw.tolist(), throttle.tolist())):
            t = t / 9.8 * hover_thr 
            y = normalize_angle(y)
            print(f"Final throttle for {agent_names[i]}: {t}")
            client.moveByRollPitchYawThrottleAsync(r, p, y, t, 0.25, agent_names[i]) 


if __name__ == '__main__':
    print("start experiments")
    main()

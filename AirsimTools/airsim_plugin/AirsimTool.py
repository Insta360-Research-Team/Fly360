import argparse
import threading
import msgpackrpc
from pathlib import Path
import glob
import time
import os
import json
import sys
import subprocess
import errno
import signal
import copy
import socket
import time

def wait_port(ip, port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket()
        s.settimeout(1)
        try:
            s.connect((ip, port))
            s.close()
            return True
        except:
            time.sleep(1)
    return False
import copy

AIRSIM_SETTINGS_TEMPLATE = {
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
    "SettingsVersion": 1.2,
    "SimMode": "ComputerVision",   # ComputerVision / Multirotor
    "ViewMode": "NoDisplay",       # Fpv / NoDisplay
    "ClockSpeed": 1,
    "CameraDefaults": {
        "CaptureSettings": [
            {"ImageType": 1, "Width": 48, "Height": 36, "FOV_Degrees": 79},
            {"ImageType": 3, "FOV_Degrees": 79}
        ]
    },
    "Recording": {"RecordInterval": 0.001, "Enabled": False, "Cameras": []},
    "SubWindows": [],
    "Vehicles": {}
}

def _cube_cameras(width=224, height=224, fov_deg=90):
    """
    6 面体相机，朝向定义：
      front: yaw   0
      back : yaw 180
      right: yaw  90
      left : yaw -90
      up   : pitch -90   (仰视)
      down : pitch  90   (俯视)
    """
    cap = [
        {"ImageType": 0, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # Scene
        {"ImageType": 1, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # DepthPlanar
        {"ImageType": 2, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # Segmentation
        {"ImageType": 3, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # DepthPerspective
    ]
    return {
        # 朝前
        "cube_front": {"CaptureSettings": cap, "X": 0.285, "Y": 0, "Z": 0, "Pitch": 0,  "Roll": 0, "Yaw":   0},
        # 朝后
        "cube_back" : {"CaptureSettings": cap, "X": -0.285, "Y": 0, "Z": 0, "Pitch": 0,  "Roll": 0, "Yaw": 180},
        # 朝右
        "cube_right": {"CaptureSettings": cap, "X": 0, "Y": 0.285, "Z": 0, "Pitch": 0,  "Roll": 0, "Yaw":  90},
        # 朝左
        "cube_left" : {"CaptureSettings": cap, "X": 0, "Y": -0.285, "Z": 0, "Pitch": 0,  "Roll": 0, "Yaw": -90},
        # 朝上（仰视）
        "cube_up"   : {"CaptureSettings": cap, "X": 0, "Y": 0, "Z": -0.285, "Pitch": 90,"Roll": 0, "Yaw":   0},
        # 朝下（俯视）
        "cube_down" : {"CaptureSettings": cap, "X": 0, "Y": 0, "Z": 0.285, "Pitch": -90, "Roll": 0, "Yaw":   0},
    }

def _high_view_camera(width=4096, height=2048, fov_deg=90):
    """
    新增高视角摄像头，分辨率 8K，位置相对 6 面体的其他摄像头更高
    """
    cap = [
        {"ImageType": 0, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # Scene
        {"ImageType": 1, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # DepthPlanar
        {"ImageType": 2, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # Segmentation
        {"ImageType": 3, "Width": width, "Height": height, "FOV_Degrees": fov_deg},  # DepthPerspective
    ]
    return {
        # 新增的高视角摄像头
        "high_view": {"CaptureSettings": cap, "X": 0, "Y": 0, "Z": -50, "Pitch": -45, "Roll": 0, "Yaw": 0},  # 高于六面体，俯视
        "right_view": {"CaptureSettings": cap, "X": -1000, "Y": -10, "Z": -35, "Pitch": -10, "Roll": 0, "Yaw": 0}  # 高于六面体，俯视
    }

def create_drones(drone_num_per_env=1, show_scene=False, uav_mode=False) -> dict:
    airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)

    airsim_settings['ViewMode'] = 'Fpv' if show_scene else 'NoDisplay'
    airsim_settings['SimMode'] = 'Multirotor' if uav_mode else 'ComputerVision'

    for i in range(drone_num_per_env):
        drone_name = f'Drone_{i+1}'
        drone = {
            "VehicleType": "SimpleFlight" if uav_mode else "ComputerVision",
            "AutoCreate": True,
            "Cameras": {**_cube_cameras(width=512, height=512, fov_deg=90), **_high_view_camera(width=4096+2048, height=2048, fov_deg=150)}  # ← 六面体 + 高视角摄像头
        }
        airsim_settings['Vehicles'][drone_name] = drone

    return airsim_settings

# =================================================================================================================================

def pid_exists(pid) -> bool:
    """
    Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def FromPortGetPid(port: int):
    subprocess_execute = "netstat -nlp | grep {}".format(
        port,
    )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    pid = None
    for line in iter(p.stdout.readline, b''):
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid


def KillPid(pid) -> None:
    if pid is None or not isinstance(pid, int):
        print('pid is not int')
        return

    while pid_exists(pid):
        try:
            # os.system(("kill -9 {}".format(pid)))
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            pass
        time.sleep(0.5)

    return


def KillPorts(ports) -> None:
    threads = []

    def _kill_port(index, port):
        pid = FromPortGetPid(port)
        KillPid(pid)

    for index, port in enumerate(ports):
        thread = threading.Thread(target=_kill_port, args=(index, port))
        threads.append(thread)
    for thread in threads:
        thread.setDaemon(True)
        thread.start()
    for thread in threads:
        thread.join()
    threads = []

    return


def KillAirVLN() -> None:
    subprocess_execute = "pkill -9 AirVLN"

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'KillAirVLN',
                e,
            )
        )
        return
    except:
        return

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    time.sleep(1)
    return


class EventHandler(object):
    def __init__(self):
        scene_ports = []
        for i in range(1000):
            scene_ports.append(
                int(args.port) + (i+1)
            )
        self.scene_ports = scene_ports

        scene_gpus = []
        while len(scene_gpus) < 100:
            scene_gpus += GPU_IDS.copy()
        self.scene_gpus = scene_gpus

        self.scene_used_ports = []

    def ping(self) -> bool:
        return True

    def _open_scenes(self, ip: str , scen_ids: list):
        print(
            "{}\tSTART closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        KillPorts(self.scene_used_ports)
        self.scene_used_ports = []
        # KillAirVLN()
        print(
            "{}\tEND closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )


        # Occupied airsim port 1
        ports = []
        index = 0
        while len(ports) < len(scen_ids):
            pid = FromPortGetPid(self.scene_ports[index])
            if pid is None or not isinstance(pid, int):
                ports.append(self.scene_ports[index])
            index += 1

        KillPorts(ports)


        # Occupied GPU 2
        gpus = [self.scene_gpus[index] for index in range(len(scen_ids))]


        # search scene path 3
        choose_env_exe_paths = []
        for scen_id in scen_ids:
            if str(scen_id).lower() == 'none':
                choose_env_exe_paths.append(None)
                continue

            res = glob.glob((str(SEARCH_ENVs_PATH) + '/**/' + 'env_' + str(scen_id) + '/LinuxNoEditor/AirVLN.sh'), recursive=True)
            if len(res) > 0:
                choose_env_exe_paths.append(res[0])
            else:
                print(f'can not find scene file: {scen_id}')
                raise KeyError


        p_s = []
        for index in range(len(scen_ids)):
            # airsim settings 4
            airsim_settings = create_drones(uav_mode=True, show_scene=False)
            airsim_settings['ApiServerPort'] = int(ports[index])
            airsim_settings['ViewMode'] = 'Fpv'  
            airsim_settings_write_content = json.dumps(airsim_settings)
            if not os.path.exists(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1))):
                os.makedirs(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1)), exist_ok=True)
            with open(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'), 'w', encoding='utf-8') as dump_f:
                dump_f.write(airsim_settings_write_content)

            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue
            else:
                subprocess_execute = "bash {} -NoSound -NoVSync -GraphicsAdapter={} --settings {} ".format(
                    choose_env_exe_paths[index],
                    gpus[index],
                    str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'),
                )

                try:
                    p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        shell=True,
                    )
                    p_s.append(p)
                except Exception as e:
                    print(
                        "{}\t{}".format(
                            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                            e,
                        )
                    )
                    return False, None
                except:
                    return False, None
        time.sleep(3)

        # check
        threads = []

        def _check_scene(index, p):
            if p is None:
                print(
                    "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                        str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        index,
                        None,
                        gpus[index],
                    )
                )
                return

            # for line in iter(p.stdout.readline, b''):
            #     if 'Drone_' in str(line):
            #         break
            ok = wait_port("127.0.0.1", ports[index], timeout=180)
            if not ok:
                raise RuntimeError("AirSim RPC port not ready")

            try:
                p.terminate()
                # os.system(("kill -9 {}".format(p.pid)))
                os.kill(p.pid, signal.SIGKILL)
            except:
                pass

            print(
                "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                    index,
                    scen_ids[index],
                    gpus[index],
                )
            )
            return

        for index, p in enumerate(p_s):
            thread = threading.Thread(target=_check_scene, args=(index, p))
            threads.append(thread)
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        threads = []

        # ChangeNice(ports)

        self.scene_used_ports += copy.deepcopy(ports)

        return True, (ip, ports)

    def reopen_scenes(self, ip: str, scen_ids: list):
        print(
            "{}\tSTART reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        try:
            result = self._open_scenes(ip, scen_ids)
        except Exception as e:
            print(e)
            result = False, None
        print(
            "{}\tEND reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result

    def close_scenes(self, ip: str) -> bool:
        print(
            "{}\tSTART close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        try:
            KillPorts(self.scene_used_ports)
            self.scene_used_ports = []
            # KillPorts(self.scene_ports)
            # KillAirVLN()

            result = True
        except Exception as e:
            print(e)
            result = False

        print(
            "{}\tEND close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result


def serve_background(server, daemon=False):
    def _start_server(server):
        server.start()
        server.close()

    t = threading.Thread(target=_start_server, args=(server,))
    t.setDaemon(daemon)
    t.start()
    return t


def serve(daemon=False):
    try:
        server = msgpackrpc.Server(EventHandler())
        addr = msgpackrpc.Address(HOST, PORT)
        server.listen(addr)

        thread = serve_background(server, daemon)

        return addr, server, thread
    except Exception as err:
        print(err)
        pass


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help='server port'
    )
    args = parser.parse_args()


    HOST = '127.0.0.1'
    PORT = int(args.port)

    CWD_DIR = Path(str(os.getcwd())).resolve()
    PROJECT_ROOT_DIR = CWD_DIR.parent
    SEARCH_ENVs_PATH = PROJECT_ROOT_DIR / 'ENVs'
    assert os.path.exists(str(SEARCH_ENVs_PATH)), 'error'

    gpu_list = []
    gpus = str(args.gpus).split(',')
    for gpu in gpus:
        gpu_list.append(int(gpu.strip()))
    GPU_IDS = gpu_list.copy()


    addr, server, thread = serve()
    print(f"start listening \t{addr._host}:{addr._port}")


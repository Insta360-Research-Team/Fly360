import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from unik3d.models import UniK3D
from unik3d.utils.camera import (MEI, OPENCV, BatchCamera, Fisheye624, Pinhole,
                                 Spherical)
from unik3d.utils.visualization import colorize, save_file_ply

SAVE = True
BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "demo"
)


def infer(model, rgb_path, camera_path, rays=None):
    rgb = np.array(Image.open(rgb_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    print(rgb_torch.shape)
    # exit()

    camera = None
    if camera_path is not None:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)

        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)

    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=rays)

    return rgb_torch, outputs


def infer_equirectangular(model, rgb_path):
    rgb = np.array(Image.open(rgb_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    # assuming full equirectangular image horizontally
    H, W = rgb.shape[:2]
    hfov_half = np.pi
    vfov_half = np.pi * H / W
    assert vfov_half <= np.pi / 2

    params = [W, H, hfov_half, vfov_half]
    camera = Spherical(params=torch.tensor([1.0] * 4 + params))

    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True)
    return rgb_torch, outputs


def save(rgb, outputs, name, base_path, save_pointcloud=False):
    depth = outputs["distance"]
    print(depth)
    import pdb; pdb.set_trace()
    rays = outputs["rays"]
    points = outputs["points"]

    depth = depth.cpu().numpy()
    rays = ((rays + 1) * 127.5).clip(0, 255)

    Image.fromarray(colorize(depth.squeeze())).save(
        os.path.join(base_path, f"{name}_depth.png")
    )
    Image.fromarray(rgb.squeeze().permute(1, 2, 0).cpu().numpy()).save(
        os.path.join(base_path, f"{name}_rgb.png")
    )
    Image.fromarray(rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()).save(
        os.path.join(base_path, f"{name}_rays.png")
    )

    if save_pointcloud:
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))


def depth_infer(model,erp_rgb_tensor,camera):

    # Trade-off between speed and resolution
    model.resolution_level = 1
    camera = Spherical(params=torch.tensor([1.0] * 4 + camera))
    outputs = model.infer(rgb=erp_rgb_tensor, camera=camera, normalize=True, rays=None)
    # 
    return outputs['distance']


def depth_infer_cube(model,erp_rgb_tensor,camera):

    # Trade-off between speed and resolution
    model.resolution_level = 1
    # camera = Spherical(params=torch.tensor([1.0] * 4 + camera))
    outputs = model.infer(rgb=erp_rgb_tensor, camera=camera, normalize=True, rays=None)
    # 
    return outputs['distance']


import time
if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    type_ = "s"  # available types: s, b, l
    name = f"unik3d-vit{type_}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    # set resolution level in [0,10) and output interpolation
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rgb = np.array(Image.open('/home/insta360/AirVLN_ws_erp/screenshot-20250910-222620.png'))
    rgb = rgb[:,:,:3]
    print(rgb.shape)
    print(rgb)
    # exit()
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    # print(rgb_torch.shape)
    # exit()
    x1 = time.time()

    result = depth_infer(model,rgb_torch,[896.0, 448.0, 3.14159, 3.14159/2])
    depth_np = result.squeeze().detach().cpu().numpy().astype(np.float32)  # (H,W)
    dmin, dmax = depth_np.min(), depth_np.max()
    print(dmin, dmax)
    print(time.time()-x1)

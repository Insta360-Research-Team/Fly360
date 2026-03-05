
import math
import torch
import torch.nn.functional as F
from torch import nn

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def _to_2tuple(x):
    if isinstance(x, tuple):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def add_lonlat_channels(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    u = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    v = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    theta = (u / W) * (2 * torch.pi) - torch.pi
    phi = (v / H) * torch.pi - (torch.pi / 2)

    th = theta[None, None, None, :].expand(B, 1, H, W)
    ph = phi[None, None, :, None].expand(B, 1, H, W)

    sin_th, cos_th = torch.sin(th), torch.cos(th)
    sin_ph, cos_ph = torch.sin(ph), torch.cos(ph)
    return torch.cat([x, sin_th, cos_th, sin_ph, cos_ph], dim=1)


# -------------------------------------------------------------
# ERP Grid Generator
# -------------------------------------------------------------
class ERPGridGenerator:
    def __init__(self, kernel_size=(3, 3), stride=(1, 1),
                 alpha: float = 0.8, eps: float = 1e-3, max_scale: float = 4.0):
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.max_scale = float(max_scale)

    def create_sampling_grid(self, H: int, W: int, device=None, dtype=None) -> torch.Tensor:
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        if Kh % 2 == 0 or Kw % 2 == 0:
            raise ValueError(f"SphereConv2d expects odd kernel_size, got ({Kh},{Kw}).")

        H_out = (H + Sh - 1) // Sh
        W_out = (W + Sw - 1) // Sw
        off_h = torch.arange(-(Kh // 2), Kh // 2 + 1, device=device)
        off_w = torch.arange(-(Kw // 2), Kw // 2 + 1, device=device)

        lat_idx = torch.empty(H_out, W_out, Kh, Kw, device=device, dtype=dtype)
        lon_idx = torch.empty(H_out, W_out, Kh, Kw, device=device, dtype=dtype)

        i_all = torch.arange(0, H, device=device)
        phi_all = ((i_all.to(dtype) + 0.5) / H) * math.pi - (math.pi / 2.0)
        cos_all = torch.cos(phi_all).abs().clamp_min(self.eps)

        for m in range(H_out):
            ih_center = min(m * Sh, H - 1)
            row_candidates = (ih_center + off_h).clamp(0, H - 1)
            cos_rows = cos_all[row_candidates.long()]
            d_w_rows = torch.clamp(self.alpha / cos_rows, max=self.max_scale)

            for n in range(W_out):
                jw_center = (n * Sw) % W
                for ai in range(Kh):
                    i_row = int(row_candidates[ai].item())
                    dw = d_w_rows[ai].item()
                    cols = (jw_center + dw * off_w).round() % W
                    lat_idx[m, n, ai, :] = float(i_row)
                    lon_idx[m, n, ai, :] = cols.to(dtype)

        y = (2.0 * (lat_idx + 0.5) / H) - 1.0
        x = (2.0 * (lon_idx + 0.5) / W) - 1.0
        grid = torch.stack([x, y], dim=-1).reshape(H_out * Kh, W_out * Kw, 2)
        return grid


# -------------------------------------------------------------
# SphereConv2d
# -------------------------------------------------------------
class SphereConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1),
                 alpha=0.8, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=_to_2tuple(kernel_size),
                         stride=_to_2tuple(stride), padding=0, dilation=1,
                         groups=groups, bias=bias)
        self._grid_shape_key = None
        self.register_buffer('grid', None, persistent=False)
        self.gridgen = ERPGridGenerator(kernel_size=self.kernel_size, stride=self.stride, alpha=alpha)

    @staticmethod
    def _dtype_code(t: torch.dtype) -> int:
        if t == torch.float16:
            return 16
        if t == torch.bfloat16:
            return 17
        return 32

    def _maybe_build_grid(self, H, W, device, dtype):
        key = (H, W, device.index if hasattr(device, 'index') else -1, self._dtype_code(dtype))
        if (self._grid_shape_key != key) or (self.grid is None) or (self.grid.device != device) or (self.grid.dtype != dtype):
            self.grid = self.gridgen.create_sampling_grid(H, W, device=device, dtype=dtype)
            self._grid_shape_key = key

    def forward(self, x):
        B, C, H, W = x.shape
        self._maybe_build_grid(H, W, x.device, x.dtype)
        grid = self.grid[None, ...].expand(B, *self.grid.shape)
        x_unfold = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
       
        feat = F.conv2d(x_unfold, self.weight, self.bias, stride=1, groups=self.groups)
        out = F.avg_pool2d(feat, kernel_size=self.kernel_size, stride=self.kernel_size)
        return out


# -------------------------------------------------------------
# Model with improved SphereConv2d
# -------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4, input_channels=6,
                 use_lonlat=False, alpha=0.8):
        super().__init__()
        self.use_lonlat = use_lonlat
        ch_in = input_channels + (4 if use_lonlat else 0)

        self.stem0 = nn.Sequential(
            SphereConv2d(ch_in, 32, (3,3), (2,2), alpha=alpha, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.05),
            SphereConv2d(32, 64, (3,3), (2,2), alpha=alpha, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
        )

        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
        )

        self.flatten = nn.Flatten()
        self.proj = nn.Linear(128 * 2 * 10, 256, bias=False)
        self.v_proj = nn.Linear(dim_obs, 256)
        nn.init.xavier_normal_(self.proj.weight, gain=0.8)
        nn.init.xavier_normal_(self.v_proj.weight, gain=0.5)

        self.gru = nn.GRUCell(256, 256)
        self.fc = nn.Linear(256, dim_action, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x, v, hx=None):
        if self.use_lonlat:
            x = add_lonlat_channels(x)
        x = self.stem0(x)
        x = self.stem1(x)
        img_feat = self.proj(self.flatten(x))
        fused = self.act(img_feat + self.v_proj(v))
        hx = self.gru(fused, hx)
        act = self.fc(self.act(hx))
        return act, None, hx

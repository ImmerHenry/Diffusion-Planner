import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Grid parameters (egocentric)
GRID_SIZE = 400  # H=W=400
RESOLUTION_M_PER_PX = 0.25  # 0.25 m/px => 100m x 100m coverage
HISTORY_FRAMES = 21
FUTURE_FRAMES = 30


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = HISTORY_FRAMES, out_channels: int = FUTURE_FRAMES, base: int = 32):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        d4 = self.dec4(torch.cat([u4, c4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        out = self.out_conv(d1)  # [B, FUTURE_FRAMES, H, W]
        return out


_unet_model: Optional[UNet2D] = None
_unet_device: Optional[torch.device] = None


def _get_unet(inputs: dict, device: torch.device) -> UNet2D:
    global _unet_model, _unet_device
    if _unet_model is None:
        _unet_model = UNet2D(in_channels=HISTORY_FRAMES, out_channels=FUTURE_FRAMES, base=32).to(device)
        ckpt_path = inputs.get("grid_guidance_unet_ckpt", None)
        if ckpt_path:
            try:
                state = torch.load(ckpt_path, map_location=device)
                # Support both raw state_dict and {'state_dict': ...}
                sd = state.get("state_dict", state)
                _unet_model.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[grid_conv_guidance] Failed to load UNet checkpoint from {ckpt_path}: {e}")
        _unet_model.eval()
        _unet_device = device
    elif _unet_device != device:
        _unet_model = _unet_model.to(device)
        _unet_device = device
    return _unet_model


def _xy_to_norm_grid(x_forward_m: torch.Tensor, y_left_m: torch.Tensor) -> torch.Tensor:
    """
    Map egocentric (x_forward, y_left) in meters to grid_sample normalized coords in [-1, 1].
    -1 is left/top, +1 is right/bottom. Grid origin (0,0) is at grid center.
    """
    half_extent_m_x = (GRID_SIZE - 1) * RESOLUTION_M_PER_PX / 2.0
    half_extent_m_y = (GRID_SIZE - 1) * RESOLUTION_M_PER_PX / 2.0

    # Horizontal (cols): -1 (left) .. +1 (right). y_left positive => move to left => negative normalized x
    norm_x = - y_left_m / half_extent_m_y
    # Vertical (rows): -1 (top) .. +1 (bottom). x_forward positive => toward top => negative normalized y
    norm_y = - x_forward_m / half_extent_m_x

    grid = torch.stack([norm_x, norm_y], dim=-1)
    return torch.clamp(grid, -1.0, 1.0)


def _scatter_points_to_map(points_px_hw: torch.Tensor, batch_size: int, channels: int) -> torch.Tensor:
    """
    points_px_hw: [B, N, 2] integer (h, w) pixel coordinates within [0, GRID_SIZE-1]
    Returns a one-hot occupancy map [B, C=channels, H, W] with points set to 1 in each channel index 0..channels-1
    Here we assume N is the number of points for a given (B, channel), arranged accordingly by caller.
    """
    device = points_px_hw.device
    H = W = GRID_SIZE
    maps = torch.zeros((batch_size, channels, H, W), device=device)
    # Flatten for scatter
    # The caller ensures that points are grouped by channel; we just clamp and set
    h = torch.clamp(points_px_hw[..., 0], 0, H - 1)
    w = torch.clamp(points_px_hw[..., 1], 0, W - 1)
    idx = h * W + w  # [B, N]
    maps = maps.view(batch_size, channels, H * W)
    # We split N into channels chunks evenly by caller; for safety, use scatter_add with a per-channel mask.
    # This function is a simple helper; we'll implement per-channel scatter in the caller for clarity.
    return maps.view(batch_size, channels, H, W)


def _rasterize_history(inputs: dict, device: torch.device) -> torch.Tensor:
    """
    Build egocentric raster input from neighbor history as [B, HISTORY_FRAMES, H, W].
    We splat agent centers per timestep as binary points and optionally blur.
    """
    neighbors: torch.Tensor = inputs["neighbor_agents_past"]  # [B, Pn, T_hist, D]
    B, Pn, T_hist, D = neighbors.shape
    T_use = min(T_hist, HISTORY_FRAMES)

    # Use last T_use frames from history
    centers = neighbors[:, :Pn, -T_use:, :2].to(device)  # [B, Pn, T_use, 2] (x_forward, y_left) in meters

    # Convert to pixel indices (h, w)
    # h = H/2 - x/RES, w = W/2 - (-y)/RES? We choose h = center - x/RES, w = center - y/RES (due to y_left positive is left)
    H = W = GRID_SIZE
    center = (GRID_SIZE - 1) / 2.0
    h = center - centers[..., 0] / RESOLUTION_M_PER_PX
    w = center - centers[..., 1] / RESOLUTION_M_PER_PX
    hw = torch.stack([h.round().long(), w.round().long()], dim=-1)  # [B, Pn, T_use, 2]

    # Build per-timestep occupancy channels
    occ = torch.zeros((B, T_use, H, W), device=device)
    occ_flat = occ.view(B, T_use, H * W)
    # For each t, scatter all agents
    for t in range(T_use):
        # mask invalid (zeros in neighbors may indicate padding). If pos is 0 for both x,y, we still allow; use a mask if velocity dims are zero? Keep simple: use a bounds mask only
        idx_flat = torch.clamp(hw[:, :, t, 0], 0, H - 1) * W + torch.clamp(hw[:, :, t, 1], 0, W - 1)  # [B, Pn]
        # Convert to one-hot add
        for b in range(B):
            occ_flat[b, t].scatter_add_(0, idx_flat[b], torch.ones_like(idx_flat[b], dtype=occ_flat.dtype, device=device))

    # Clip to [0,1]
    occ = torch.clamp(occ, 0.0, 1.0)

    # Optional: light blur to expand points (approximate agent footprint)
    kernel_size = 7
    sigma = 2.0
    grid = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = torch.einsum("i,j->ij", gauss_1d, gauss_1d)
    kernel = kernel[None, None, :, :].repeat(T_use, 1, 1, 1)  # depthwise per-channel
    occ = F.conv2d(
        F.pad(occ, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="replicate"),
        kernel,
        groups=T_use,
    )

    # If history shorter than required, pad channels at front
    if T_use < HISTORY_FRAMES:
        pad = torch.zeros((B, HISTORY_FRAMES - T_use, H, W), device=device)
        occ = torch.cat([pad, occ], dim=1)

    return occ  # [B, HISTORY_FRAMES, H, W]


@torch.no_grad()
def _predict_cost_map(inputs: dict, device: torch.device) -> torch.Tensor:
    """
    Returns cost logits/probabilities for FUTURE_FRAMES: [B, FUTURE_FRAMES, H, W]
    """
    raster_in = _rasterize_history(inputs, device)  # [B, C_in, H, W]
    unet = _get_unet(inputs, device)
    logits = unet(raster_in)
    probs = torch.sigmoid(logits)
    return probs


def grid_conv_guidance_fn(x: torch.Tensor, t: torch.Tensor, cond, inputs: dict, *args, **kwargs) -> torch.Tensor:
    """
    Guidance that penalizes trajectories passing through high-cost (predicted occupancy) regions.

    x: [B, Pn+1, T+1, 4] (denormalized in wrapper)
    t: [B] diffusion time
    inputs: Dict[str, Tensor] with at least 'neighbor_agents_past'. Optional 'grid_guidance_unet_ckpt'.
    """
    device = x.device
    B, P, T_plus_1, _ = x.shape

    # Only enable gradient when diffusion time is within effective range
    mask_diffusion_time = (t < 0.1) & (t > 0.005)
    x = torch.where(mask_diffusion_time[:, None, None, None], x, x.detach())

    # Use ego future positions (exclude t=0 current)
    ego_future = x[:, 0, 1:, :2]  # [B, T, 2] (x_forward, y_left)
    T_future = ego_future.shape[1]

    # Predict cost map (no grad)
    cost_map = _predict_cost_map(inputs, device)  # [B, FUTURE_FRAMES, H, W]

    # Sample costs along the trajectory using bilinear sampling
    T_use = min(T_future, FUTURE_FRAMES)
    ego_use = ego_future[:, :T_use, :]  # [B, T_use, 2]

    # Compute normalized coords for grid_sample: [B, T_use, 2]
    norm_grid = _xy_to_norm_grid(ego_use[..., 0], ego_use[..., 1])  # [B, T_use, 2]

    # Accumulate sampled costs
    sampled_costs = []
    for step in range(T_use):
        # input: [B, 1, H, W]; grid: [B, 1, 1, 2]
        cm_step = cost_map[:, step:step + 1, :, :]
        grid_step = norm_grid[:, step:step + 1, :].unsqueeze(1).unsqueeze(2)
        val = F.grid_sample(cm_step, grid_step, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled_costs.append(val[:, 0, 0, 0])  # [B]

    costs = torch.stack(sampled_costs, dim=1)  # [B, T_use]

    # Time weighting (optional: emphasize near future slightly)
    time_weights = torch.linspace(1.0, 1.5, T_use, device=device)
    weighted_cost = (costs * time_weights[None, :]).sum(dim=1) / time_weights.sum()

    # We want to guide toward low-cost regions => reward = - average cost
    reward = -weighted_cost  # [B]

    # Backprop through x by introducing an auxiliary dot; similar style as collision guidance
    # Promote gradient flow to ego positions
    reward_sum = reward.sum()
    x_aux = torch.autograd.grad(reward_sum, x, retain_graph=True, allow_unused=True)[0]

    # If grad is None due to mask, fallback to zero
    if x_aux is None:
        return reward  # already [B]

    # Only use position components for ego and future steps; zero others
    x_aux_pos = torch.zeros_like(x)
    x_aux_pos[:, 0, 1:, :2] = x_aux[:, 0, 1:, :2]

    # Project auxiliary gradient back to a scalar reward to be used by classifier guidance
    projected = (x_aux_pos[:, 0, 1:, :2] * x[:, 0, 1:, :2]).sum(dim=(1, 2))  # [B]

    # Combine original reward and projected term for stronger signal
    final_reward = reward + 0.5 * projected

    return final_reward
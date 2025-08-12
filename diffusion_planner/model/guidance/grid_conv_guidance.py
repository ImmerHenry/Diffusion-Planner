import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = [
    "OccupancyUNet",
    "grid_conv_guidance_fn",
]


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, D403
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, D403
        return self.down(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # noqa: D401, D403
        x1 = self.up(x1)
        # Pad x1 to the size of x2 if needed (should not happen for power of 2 spatial dims)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, D403
        return self.conv(x)


class OccupancyUNet(nn.Module):
    """Simple UNet tailored for occupancy cost-map prediction.

    Inputs:
        history (B, 21, H, W): Raster of historical occupancy likelihoods (egocentric).
    Returns:
        future_cost (B, 30, H, W): Predicted per-pixel cost for each of 30 future steps.
    """

    def __init__(self, in_channels: int = 21, out_channels: int = 30, base_c: int = 64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 8)
        self.up1 = Up(base_c * 16, base_c * 4)
        self.up2 = Up(base_c * 8, base_c * 2)
        self.up3 = Up(base_c * 4, base_c)
        self.up4 = Up(base_c * 2, base_c)
        self.outc = OutConv(base_c, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, D403
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)  # (B, 30, H, W)
        # We interpret raw logits as cost; optional sigmoid depending on training.
        return x


def _positions_to_grid(positions: torch.Tensor, h: int, w: int, resolution: float) -> torch.Tensor:
    """Convert egocentric positions in metres to normalized grid coordinates [-1, 1] for grid_sample.

    Args:
        positions: (B, T, 2) tensor – x forward (m), y left (m).
        h, w: spatial size of the raster.
        resolution: metres per pixel.

    Returns:
        grid: (B*T, 1, 1, 2) tensor suitable for F.grid_sample where last dim order is (x_normalised, y_normalised).
    """
    # Convert metres to pixel index (origin at centre)
    px = positions[..., 0] / resolution + (w - 1) / 2.0
    py = -positions[..., 1] / resolution + (h - 1) / 2.0  # y left => -y for row indexing (row 0 at top)
    # Normalise to [-1,1]
    nx = px / (w - 1) * 2 - 1
    ny = py / (h - 1) * 2 - 1
    grid = torch.stack([nx, ny], dim=-1)  # (B, T, 2)
    grid = grid.view(-1, 1, 1, 2)  # (B*T, 1, 1, 2)
    return grid


def grid_conv_guidance_fn(
    x: torch.Tensor,
    t: torch.Tensor,
    cond,  # unused but kept for interface compatibility
    *,
    inputs: dict,
    occupancy_unet: OccupancyUNet,
    cost_resolution: float = 0.25,
    cost_weight: float = 1.0,
) -> torch.Tensor:
    """Guidance that pulls the trajectory towards low-cost regions of a raster cost-map.

    The cost-map is produced by an egocentric UNet taking the past 21 frames of occupancy and predicting 30 future frames.

    Args:
        x: (B, P, T+1, 4) current trajectory samples in the diffusion process. We only consider the ego agent (index 0).
        t: Diffusion time step tensor (B,).
        inputs: dict containing at least a "history_occupancy" key with shape (B, 21, H, W).
        occupancy_unet: Instance of OccupancyUNet with loaded weights.
        cost_resolution: metres per pixel of the raster (default 0.25).
        cost_weight: scaling factor for the guidance magnitude.

    Returns:
        Scalar energy per sample (B,) – higher values for trajectories in high-cost regions.
    """
    assert "history_occupancy" in inputs, "history_occupancy must be provided in inputs for grid_conv_guidance_fn"

    B, P, TP1, _ = x.shape
    T = TP1 - 1  # future horizon (should be 30)
    device = x.device

    # 1. Predict cost maps with UNet
    history_occ = inputs["history_occupancy"].to(device)  # (B, 21, H, W)
    cost_maps = occupancy_unet(history_occ)  # (B, 30, H, W)

    # Optional: apply softplus/sigmoid to ensure positivity – treat raw logits as cost for now
    cost_maps = F.softplus(cost_maps)  # ensure non-negative cost

    _, _, H, W = cost_maps.shape

    # 2. Positions of ego over future horizon (exclude current step 0)
    ego_positions = x[:, 0, 1:, :2]  # (B, T, 2)

    # Make sure we can compute gradients w.r.t positions
    mask_diffusion_time = (t < 0.1) & (t > 0.005)
    ego_positions = torch.where(mask_diffusion_time[:, None, None], ego_positions, ego_positions.detach())

    # 3. Sample cost along trajectory
    grid = _positions_to_grid(ego_positions, h=H, w=W, resolution=cost_resolution).to(device)
    cost_maps_flat = cost_maps.view(B * T, 1, H, W)
    sampled_cost = F.grid_sample(cost_maps_flat, grid, align_corners=True, mode="bilinear").view(B, T)

    # 4. Energy is average cost along the path – higher cost => higher energy
    energy_raw = sampled_cost.mean(dim=1)  # (B,)

    # 5. Multiply by weight and return (gradient will flow to positions through grid_sample)
    return cost_weight * energy_raw
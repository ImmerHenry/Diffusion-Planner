import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


# Grid configuration constants
GRID_SIZE = 400  # 400x400 grid
GRID_RESOLUTION = 0.25  # 0.25 meters per pixel
GRID_RANGE = GRID_SIZE * GRID_RESOLUTION  # 100 meters total range
HISTORY_FRAMES = 21
FUTURE_FRAMES = 30


class UNetOccupancyPredictor(nn.Module):
    """
    UNET model for occupancy prediction from historical frames to future frames.
    Input: 21 historical frames (ego-centric)
    Output: 30 future frames occupancy prediction (ego-centric)
    """
    def __init__(self, in_channels=HISTORY_FRAMES, out_channels=FUTURE_FRAMES):
        super(UNetOccupancyPredictor, self).__init__()
        
        # Encoder
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, out_channels, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output with sigmoid for occupancy probability
        output = torch.sigmoid(self.final(dec1))
        
        return output


def world_to_grid_coords(world_coords: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
    """
    Convert world coordinates to ego-centric grid coordinates.
    
    Args:
        world_coords: [B, N, 2] world coordinates (x, y)
        ego_pose: [B, 3] ego pose (x, y, heading)
    
    Returns:
        grid_coords: [B, N, 2] grid coordinates (i, j)
    """
    B, N, _ = world_coords.shape
    
    # Extract ego position and heading
    ego_x, ego_y, ego_heading = ego_pose[:, 0], ego_pose[:, 1], ego_pose[:, 2]
    
    # Transform to ego-centric coordinates
    cos_h, sin_h = torch.cos(ego_heading), torch.sin(ego_heading)
    
    # Relative coordinates
    rel_x = world_coords[:, :, 0] - ego_x.unsqueeze(1)
    rel_y = world_coords[:, :, 1] - ego_y.unsqueeze(1)
    
    # Rotate to ego frame
    ego_x_coords = cos_h.unsqueeze(1) * rel_x + sin_h.unsqueeze(1) * rel_y
    ego_y_coords = -sin_h.unsqueeze(1) * rel_x + cos_h.unsqueeze(1) * rel_y
    
    # Convert to grid coordinates (center of grid is ego position)
    grid_center = GRID_SIZE // 2
    grid_i = grid_center - ego_y_coords / GRID_RESOLUTION
    grid_j = grid_center + ego_x_coords / GRID_RESOLUTION
    
    return torch.stack([grid_i, grid_j], dim=-1)


def sample_trajectory_on_grid(trajectory: torch.Tensor, ego_pose: torch.Tensor, 
                             cost_map: torch.Tensor) -> torch.Tensor:
    """
    Sample trajectory points on the cost grid and compute costs.
    
    Args:
        trajectory: [B, T, 2] trajectory in world coordinates
        ego_pose: [B, 3] current ego pose (x, y, heading)
        cost_map: [B, FUTURE_FRAMES, GRID_SIZE, GRID_SIZE] cost maps for future frames
    
    Returns:
        costs: [B, T] cost values along the trajectory
    """
    B, T, _ = trajectory.shape
    
    # Convert trajectory to grid coordinates
    grid_coords = world_to_grid_coords(trajectory, ego_pose)  # [B, T, 2]
    
    # Clamp coordinates to valid grid range
    grid_coords = torch.clamp(grid_coords, 0, GRID_SIZE - 1)
    
    # Sample costs from the grid
    costs = torch.zeros(B, T, device=trajectory.device)
    
    for t in range(min(T, FUTURE_FRAMES)):
        # Use bilinear interpolation for sampling
        grid_i, grid_j = grid_coords[:, t, 0], grid_coords[:, t, 1]
        
        # Get integer and fractional parts
        i0 = torch.floor(grid_i).long()
        i1 = torch.clamp(i0 + 1, 0, GRID_SIZE - 1)
        j0 = torch.floor(grid_j).long()
        j1 = torch.clamp(j0 + 1, 0, GRID_SIZE - 1)
        
        di = grid_i - i0.float()
        dj = grid_j - j0.float()
        
        # Bilinear interpolation
        cost_00 = cost_map[torch.arange(B), t, i0, j0]
        cost_01 = cost_map[torch.arange(B), t, i0, j1]
        cost_10 = cost_map[torch.arange(B), t, i1, j0]
        cost_11 = cost_map[torch.arange(B), t, i1, j1]
        
        cost = (cost_00 * (1 - di) * (1 - dj) +
                cost_01 * (1 - di) * dj +
                cost_10 * di * (1 - dj) +
                cost_11 * di * dj)
        
        costs[:, t] = cost
    
    return costs


def create_mock_unet_predictor(device: torch.device) -> UNetOccupancyPredictor:
    """
    Create a mock UNET predictor for demonstration.
    In practice, this would be loaded from a trained checkpoint.
    """
    model = UNetOccupancyPredictor().to(device)
    model.eval()
    return model


def grid_conv_guidance_fn(x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, 
                         inputs: Dict, *args, **kwargs) -> torch.Tensor:
    """
    Grid convolution guidance function that guides trajectories towards low-cost regions
    based on UNET-generated occupancy prediction.
    
    Args:
        x: [B, P, T*4] trajectory tensor (B: batch, P: agents, T: time steps)
        t: [B] diffusion timestep
        cond: conditioning information
        inputs: dict containing input data
        
    Returns:
        reward: [B] guidance reward for each batch
    """
    B, P, _ = x.shape
    T = _ // 4  # Each time step has [x, y, cos_h, sin_h]
    
    # Reshape trajectory
    x = x.reshape(B, P, T, 4)
    
    # Only apply guidance during specific diffusion time range
    mask_diffusion_time = (t < 0.1) & (t > 0.005)
    x = torch.where(mask_diffusion_time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                   x, x.detach())
    
    # Extract ego trajectory (first agent)
    ego_traj = x[:, 0, 1:, :2]  # [B, T-1, 2] - future trajectory, x,y only
    
    # Get current ego pose for coordinate transformation
    ego_current = x[:, 0, 0, :]  # [B, 4] current ego state
    ego_pose = torch.stack([
        ego_current[:, 0],  # x
        ego_current[:, 1],  # y
        torch.atan2(ego_current[:, 3], ego_current[:, 2])  # heading from cos,sin
    ], dim=1)  # [B, 3]
    
    # Create mock historical occupancy data
    # In practice, this would come from the actual sensor data
    device = x.device
    history_occupancy = torch.rand(B, HISTORY_FRAMES, GRID_SIZE, GRID_SIZE, 
                                  device=device) * 0.3  # Low random occupancy
    
    # Create mock UNET model for demonstration
    # In practice, load from checkpoint
    if not hasattr(grid_conv_guidance_fn, '_unet_model'):
        grid_conv_guidance_fn._unet_model = create_mock_unet_predictor(device)
    
    unet_model = grid_conv_guidance_fn._unet_model
    
    # Generate future occupancy prediction
    with torch.no_grad():
        future_occupancy = unet_model(history_occupancy)  # [B, FUTURE_FRAMES, GRID_SIZE, GRID_SIZE]
    
    # Convert occupancy to cost (higher occupancy = higher cost)
    cost_map = future_occupancy.clone()
    
    # Add some cost for areas close to obstacles
    kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0
    dilated_cost = F.conv2d(cost_map.view(-1, 1, GRID_SIZE, GRID_SIZE), 
                           kernel, padding=2)
    cost_map = cost_map.view(-1, FUTURE_FRAMES, GRID_SIZE, GRID_SIZE) + \
               0.5 * dilated_cost.view(-1, FUTURE_FRAMES, GRID_SIZE, GRID_SIZE)
    
    # Sample trajectory costs
    trajectory_costs = sample_trajectory_on_grid(ego_traj, ego_pose, cost_map)  # [B, T-1]
    
    # Compute guidance reward (negative cost with exponential weighting for recent time steps)
    time_weights = torch.exp(-0.1 * torch.arange(T-1, device=device).float())
    time_weights = time_weights / time_weights.sum()
    
    weighted_cost = torch.sum(trajectory_costs * time_weights.unsqueeze(0), dim=1)  # [B]
    
    # Convert to reward (lower cost = higher reward)
    reward = -weighted_cost
    
    # Add small penalty for high-cost regions to encourage avoidance
    high_cost_penalty = torch.clamp(weighted_cost - 0.5, min=0) ** 2
    reward = reward - 2.0 * high_cost_penalty
    
    return 1.0 * reward  # Scale factor for guidance strength
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

class UNetOccupancyPredictor(nn.Module):
    """
    UNet模型用于生成occupancy prediction
    输入: 21帧历史观测 [B, 21, C, H, W]
    输出: 30帧未来occupancy [B, 30, H, W]
    """
    def __init__(self, input_channels=64, hidden_dim=128, output_frames=30):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_frames = output_frames
        
        # Encoder
        self.encoder1 = self._make_layer(input_channels, hidden_dim)
        self.encoder2 = self._make_layer(hidden_dim, hidden_dim * 2)
        self.encoder3 = self._make_layer(hidden_dim * 2, hidden_dim * 4)
        self.encoder4 = self._make_layer(hidden_dim * 4, hidden_dim * 8)
        
        # Decoder
        self.decoder4 = self._make_layer(hidden_dim * 8, hidden_dim * 4)
        self.decoder3 = self._make_layer(hidden_dim * 4, hidden_dim * 2)
        self.decoder2 = self._make_layer(hidden_dim * 2, hidden_dim)
        self.decoder1 = self._make_layer(hidden_dim, hidden_dim)
        
        # Final output layer
        self.final_conv = nn.Conv2d(hidden_dim, output_frames, kernel_size=1)
        
        # Temporal processing
        self.temporal_conv = nn.Conv3d(input_channels, hidden_dim, 
                                      kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: [B, 21, C, H, W] -> [B, C, 21, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # [B, hidden_dim, 21, H, W]
        x = x.mean(dim=2)  # [B, hidden_dim, H, W] - temporal pooling
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        # Decoder with skip connections
        d4 = self.decoder4(F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False))
        d3 = self.decoder3(F.interpolate(d4 + e3, scale_factor=2, mode='bilinear', align_corners=False))
        d2 = self.decoder2(F.interpolate(d3 + e2, scale_factor=2, mode='bilinear', align_corners=False))
        d1 = self.decoder1(F.interpolate(d2 + e1, scale_factor=2, mode='bilinear', align_corners=False))
        
        # Final output
        occupancy = self.final_conv(d1)  # [B, 30, H, W]
        
        return torch.sigmoid(occupancy)  # 0-1 occupancy probability

class ObservationToGridConverter:
    """
    将观测数据转换为栅格化表示
    """
    def __init__(self, grid_size=400, resolution=0.25):
        self.grid_size = grid_size
        self.resolution = resolution
        self.half_size = grid_size // 2
        
    def convert_agents_to_grid(self, neighbor_agents_past: torch.Tensor, 
                              ego_current_state: torch.Tensor) -> torch.Tensor:
        """
        将智能体历史轨迹转换为栅格表示
        
        Args:
            neighbor_agents_past: [B, N, T, F] 邻居智能体历史轨迹
            ego_current_state: [B, F] ego当前状态
            
        Returns:
            grid: [B, T, H, W] 栅格化表示
        """
        B, N, T, F = neighbor_agents_past.shape
        grid = torch.zeros(B, T, self.grid_size, self.grid_size, device=neighbor_agents_past.device)
        
        # 处理邻居智能体
        for b in range(B):
            for n in range(N):
                for t in range(T):
                    # 提取位置信息 (假设前两个维度是x, y)
                    x, y = neighbor_agents_past[b, n, t, 0], neighbor_agents_past[b, n, t, 1]
                    
                    # 转换为栅格索引
                    x_idx = int((x / self.resolution) + self.half_size)
                    y_idx = int((y / self.resolution) + self.half_size)
                    
                    # 确保索引在有效范围内
                    if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                        # 在智能体周围创建占用区域
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = x_idx + dx, y_idx + dy
                                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                    grid[b, t, nx, ny] = 1.0
        
        # 处理ego车辆
        for b in range(B):
            x, y = ego_current_state[b, 0], ego_current_state[b, 1]
            x_idx = int((x / self.resolution) + self.half_size)
            y_idx = int((y / self.resolution) + self.half_size)
            
            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                # 在ego周围创建占用区域
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = x_idx + dx, y_idx + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            grid[b, -1, nx, ny] = 1.0  # 只在最后一帧添加ego
        
        return grid
    
    def convert_map_to_grid(self, map_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将地图特征转换为栅格表示
        
        Args:
            map_features: 地图特征字典
            
        Returns:
            grid: [B, H, W] 地图栅格
        """
        B = 1  # 假设batch size为1
        grid = torch.zeros(B, self.grid_size, self.grid_size, device=next(iter(map_features.values())).device)
        
        # 这里可以根据实际的地图特征格式进行转换
        # 目前使用占位符
        return grid

class GridCostMapGenerator:
    """
    栅格化成本地图生成器
    将UNet输出的occupancy prediction转换为成本地图
    """
    def __init__(self, grid_size=400, resolution=0.25, cost_scale=10.0):
        self.grid_size = grid_size  # 400x400
        self.resolution = resolution  # 0.25m/pixel
        self.cost_scale = cost_scale
        
        # 创建网格坐标
        self.x_coords = torch.arange(-grid_size//2, grid_size//2) * resolution
        self.y_coords = torch.arange(-grid_size//2, grid_size//2) * resolution
        
    def generate_cost_map(self, occupancy_pred: torch.Tensor, 
                         ego_trajectory: torch.Tensor,
                         neighbor_trajectories: Optional[torch.Tensor] = None,
                         map_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成成本地图
        
        Args:
            occupancy_pred: [B, 30, H, W] UNet输出的occupancy prediction
            ego_trajectory: [B, T, 4] ego轨迹 (x, y, cos_h, sin_h)
            neighbor_trajectories: [B, N, T, 4] 邻居轨迹
            map_grid: [B, H, W] 地图栅格
            
        Returns:
            cost_map: [B, H, W] 成本地图
        """
        B, T, H, W = occupancy_pred.shape
        
        # 基础成本地图：occupancy作为障碍物成本
        cost_map = occupancy_pred.mean(dim=1) * self.cost_scale  # [B, H, W]
        
        # 添加地图成本
        if map_grid is not None:
            cost_map = cost_map + map_grid * 5.0
        
        # 添加ego轨迹成本（鼓励轨迹平滑）
        if ego_trajectory is not None:
            ego_cost = self._add_trajectory_cost(cost_map, ego_trajectory, is_ego=True)
            cost_map = cost_map + ego_cost
        
        # 添加邻居轨迹成本
        if neighbor_trajectories is not None:
            neighbor_cost = self._add_trajectory_cost(cost_map, neighbor_trajectories, is_ego=False)
            cost_map = cost_map + neighbor_cost
        
        # 添加边界成本
        boundary_cost = self._add_boundary_cost(cost_map)
        cost_map = cost_map + boundary_cost
        
        # 平滑成本地图
        cost_map = self._smooth_cost_map(cost_map)
        
        return cost_map
    
    def _add_trajectory_cost(self, cost_map: torch.Tensor, 
                            trajectory: torch.Tensor, 
                            is_ego: bool = False) -> torch.Tensor:
        """添加轨迹相关的成本"""
        B, H, W = cost_map.shape
        cost = torch.zeros_like(cost_map)
        
        # 将轨迹坐标转换为网格索引
        x_coords = trajectory[..., 0]  # [B, T] or [B, N, T]
        y_coords = trajectory[..., 1]  # [B, T] or [B, N, T]
        
        # 转换为网格索引
        x_idx = ((x_coords / self.resolution) + self.grid_size // 2).long()
        y_idx = ((y_coords / self.resolution) + self.grid_size // 2).long()
        
        # 确保索引在有效范围内
        x_idx = torch.clamp(x_idx, 0, self.grid_size - 1)
        y_idx = torch.clamp(y_idx, 0, self.grid_size - 1)
        
        # 添加轨迹成本
        if is_ego:
            # Ego轨迹：低成本路径
            for b in range(B):
                for t in range(x_idx.shape[-1]):
                    if x_idx.shape[-2] == 1:  # ego轨迹
                        i, j = x_idx[b, t], y_idx[b, t]
                        # 在ego轨迹周围创建低成本区域
                        for dx in range(-3, 4):
                            for dy in range(-3, 4):
                                ni, nj = i + dx, j + dy
                                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                                    cost[b, ni, nj] -= 1.0  # 负成本表示奖励
                    else:  # 邻居轨迹
                        for n in range(x_idx.shape[-2]):
                            i, j = x_idx[b, n, t], y_idx[b, n, t]
                            cost[b, i, j] += 1.0  # 正成本表示惩罚
        else:
            # 邻居轨迹：高成本区域
            for b in range(B):
                for n in range(x_idx.shape[-2]):
                    for t in range(x_idx.shape[-1]):
                        i, j = x_idx[b, n, t], y_idx[b, n, t]
                        # 在邻居轨迹周围创建高成本区域
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                ni, nj = i + dx, j + dy
                                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                                    cost[b, ni, nj] += 2.0  # 高成本避免碰撞
        
        return cost
    
    def _add_boundary_cost(self, cost_map: torch.Tensor) -> torch.Tensor:
        """添加边界成本"""
        B, H, W = cost_map.shape
        cost = torch.zeros_like(cost_map)
        
        # 边界成本
        boundary_width = 20  # 边界宽度（像素）
        cost[:, :boundary_width, :] += 5.0  # 上边界
        cost[:, -boundary_width:, :] += 5.0  # 下边界
        cost[:, :, :boundary_width] += 5.0  # 左边界
        cost[:, :, -boundary_width:] += 5.0  # 右边界
        
        return cost
    
    def _smooth_cost_map(self, cost_map: torch.Tensor) -> torch.Tensor:
        """平滑成本地图"""
        # 使用高斯滤波平滑成本地图
        kernel_size = 5
        sigma = 1.0
        
        # 创建高斯核
        kernel = torch.exp(-torch.arange(-kernel_size//2, kernel_size//2+1)**2 / (2*sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).to(cost_map.device)
        
        # 应用高斯滤波
        cost_map = F.conv2d(cost_map.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), 
                           padding=(0, kernel_size//2))
        cost_map = F.conv2d(cost_map, kernel.unsqueeze(0).unsqueeze(0).transpose(2, 3), 
                           padding=(kernel_size//2, 0))
        
        return cost_map.squeeze(1)

def grid_conv_guidance_fn(x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, 
                         inputs: Dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
    """
    基于栅格化成本地图的guidance函数
    
    Args:
        x: [B, P, T, 4] 轨迹 (x, y, cos_h, sin_h)
        t: [B] 时间步
        cond: 条件信息
        inputs: 输入数据字典
        
    Returns:
        energy: [B] guidance能量
    """
    B, P, T, _ = x.shape
    
    # 只在特定时间步应用guidance
    mask_diffusion_time = (t < 0.1) & (t > 0.005)
    if not mask_diffusion_time.any():
        return torch.zeros(B, device=x.device)
    
    # 分离ego和邻居轨迹
    ego_trajectory = x[:, 0, 1:, :]  # [B, T, 4] ego未来轨迹
    neighbor_trajectories = x[:, 1:, 1:, :]  # [B, P-1, T, 4] 邻居未来轨迹
    
    # 创建观测到栅格的转换器
    obs_converter = ObservationToGridConverter(grid_size=400, resolution=0.25)
    
    # 从inputs中提取观测数据
    neighbor_agents_past = inputs.get("neighbor_agents_past", None)
    ego_current_state = inputs.get("ego_current_state", None)
    
    if neighbor_agents_past is not None and ego_current_state is not None:
        # 将观测转换为栅格
        obs_grid = obs_converter.convert_agents_to_grid(neighbor_agents_past, ego_current_state)
        
        # 创建UNet occupancy predictor（这里假设已经预训练）
        # 在实际使用中，这个模型应该从checkpoint加载
        occupancy_predictor = UNetOccupancyPredictor(
            input_channels=obs_grid.shape[1],  # 使用实际的时间步数
            hidden_dim=128,
            output_frames=30
        ).to(x.device)
        
        # 生成occupancy prediction
        with torch.no_grad():
            occupancy_pred = occupancy_predictor(obs_grid)
    else:
        # 如果没有观测数据，使用占位符
        occupancy_pred = torch.randn(B, 30, 400, 400, device=x.device) * 0.1
    
    # 创建成本地图生成器
    cost_generator = GridCostMapGenerator(
        grid_size=400,
        resolution=0.25,
        cost_scale=10.0
    )
    
    # 生成成本地图
    cost_map = cost_generator.generate_cost_map(
        occupancy_pred, ego_trajectory, neighbor_trajectories
    )
    
    # 计算梯度引导
    guidance_energy = compute_guidance_energy(ego_trajectory, cost_map, cost_generator.resolution)
    
    return guidance_energy

def compute_guidance_energy(trajectory: torch.Tensor, cost_map: torch.Tensor, 
                           resolution: float) -> torch.Tensor:
    """计算guidance能量，引导轨迹去往低成本区域"""
    B, T, _ = trajectory.shape
    
    # 计算成本地图的梯度
    cost_grad_x = torch.gradient(cost_map, dim=2)[0]  # [B, H, W]
    cost_grad_y = torch.gradient(cost_map, dim=1)[0]  # [B, H, W]
    
    # 将轨迹坐标转换为网格索引
    x_coords = trajectory[..., 0]
    y_coords = trajectory[..., 1]
    
    x_idx = ((x_coords / resolution) + 200).long()
    y_idx = ((y_coords / resolution) + 200).long()
    
    # 确保索引在有效范围内
    x_idx = torch.clamp(x_idx, 0, 399)
    y_idx = torch.clamp(y_idx, 0, 399)
    
    # 采样梯度值
    batch_indices = torch.arange(B, device=trajectory.device)[:, None].expand(-1, T)
    grad_x = cost_grad_x[batch_indices, x_idx, y_idx]  # [B, T]
    grad_y = cost_grad_y[batch_indices, x_idx, y_idx]  # [B, T]
    
    # 计算梯度引导力
    guidance_force = torch.stack([grad_x, grad_y], dim=-1)  # [B, T, 2]
    
    # 计算guidance能量（负梯度方向引导）
    guidance_energy = -torch.sum(guidance_force * trajectory[..., :2], dim=(1, 2))
    
    return guidance_energy
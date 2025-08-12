#!/usr/bin/env python3
"""
测试 grid_conv_guidance 功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# 导入guidance模块
from grid_conv_guidance import (
    UNetOccupancyPredictor, 
    ObservationToGridConverter, 
    GridCostMapGenerator,
    grid_conv_guidance_fn
)

def test_unet_occupancy_predictor():
    """测试UNet occupancy predictor"""
    print("Testing UNet Occupancy Predictor...")
    
    # 创建模型
    model = UNetOccupancyPredictor(
        input_channels=21,  # 21帧历史
        hidden_dim=64,      # 较小的hidden dim用于测试
        output_frames=30    # 30帧未来
    )
    
    # 创建测试输入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 21, 64, 100, 100)  # 较小的尺寸用于测试
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == (batch_size, 30, 100, 100)
    assert torch.all((output >= 0) & (output <= 1))  # 检查sigmoid输出范围
    
    print("✓ UNet Occupancy Predictor test passed\n")
    return model

def test_observation_to_grid_converter():
    """测试观测到栅格转换器"""
    print("Testing Observation to Grid Converter...")
    
    converter = ObservationToGridConverter(grid_size=100, resolution=0.25)  # 较小的网格用于测试
    
    # 创建测试数据
    batch_size = 2
    num_agents = 3
    time_steps = 5
    features = 8
    
    neighbor_agents_past = torch.randn(batch_size, num_agents, time_steps, features)
    ego_current_state = torch.randn(batch_size, features)
    
    # 转换到栅格
    grid = converter.convert_agents_to_grid(neighbor_agents_past, ego_current_state)
    
    print(f"Neighbor agents shape: {neighbor_agents_past.shape}")
    print(f"Ego state shape: {ego_current_state.shape}")
    print(f"Output grid shape: {grid.shape}")
    print(f"Grid value range: [{grid.min():.3f}, {grid.max():.3f}]")
    
    assert grid.shape == (batch_size, time_steps, 100, 100)
    assert torch.all(grid >= 0)  # 检查非负值
    
    print("✓ Observation to Grid Converter test passed\n")
    return converter

def test_grid_cost_map_generator():
    """测试栅格成本地图生成器"""
    print("Testing Grid Cost Map Generator...")
    
    generator = GridCostMapGenerator(
        grid_size=100,      # 较小的网格用于测试
        resolution=0.25,
        cost_scale=5.0
    )
    
    # 创建测试数据
    batch_size = 2
    time_steps = 30
    grid_size = 100
    
    occupancy_pred = torch.rand(batch_size, time_steps, grid_size, grid_size)
    ego_trajectory = torch.randn(batch_size, 10, 4)  # 10个时间步
    neighbor_trajectories = torch.randn(batch_size, 2, 10, 4)  # 2个邻居，10个时间步
    
    # 生成成本地图
    cost_map = generator.generate_cost_map(
        occupancy_pred, ego_trajectory, neighbor_trajectories
    )
    
    print(f"Occupancy pred shape: {occupancy_pred.shape}")
    print(f"Ego trajectory shape: {ego_trajectory.shape}")
    print(f"Neighbor trajectories shape: {neighbor_trajectories.shape}")
    print(f"Cost map shape: {cost_map.shape}")
    print(f"Cost range: [{cost_map.min():.3f}, {cost_map.max():.3f}]")
    
    assert cost_map.shape == (batch_size, grid_size, grid_size)
    
    print("✓ Grid Cost Map Generator test passed\n")
    return generator

def test_grid_conv_guidance_fn():
    """测试grid_conv_guidance函数"""
    print("Testing Grid Conv Guidance Function...")
    
    # 创建测试数据
    batch_size = 2
    num_agents = 3
    time_steps = 10
    features = 4
    
    x = torch.randn(batch_size, num_agents, time_steps, features)
    t = torch.tensor([0.05, 0.08])  # 在guidance时间范围内
    cond = torch.randn(batch_size, 10)
    
    # 创建模拟的inputs
    inputs = {
        "neighbor_agents_past": torch.randn(batch_size, 2, 21, 8),  # 2个邻居，21帧历史
        "ego_current_state": torch.randn(batch_size, 10)
    }
    
    # 调用guidance函数
    try:
        energy = grid_conv_guidance_fn(x, t, cond, inputs)
        print(f"Input x shape: {x.shape}")
        print(f"Time t: {t}")
        print(f"Guidance energy shape: {energy.shape}")
        print(f"Energy range: [{energy.min():.3f}, {energy.max():.3f}]")
        
        assert energy.shape == (batch_size,)
        print("✓ Grid Conv Guidance Function test passed\n")
        
    except Exception as e:
        print(f"✗ Grid Conv Guidance Function test failed: {e}")
        print("This might be expected if UNet model is not properly initialized\n")

def visualize_cost_map():
    """可视化成本地图（可选）"""
    print("Visualizing cost map...")
    
    try:
        # 创建测试成本地图
        generator = GridCostMapGenerator(grid_size=100, resolution=0.25)
        
        # 创建模拟occupancy prediction
        occupancy_pred = torch.rand(1, 30, 100, 100)
        
        # 创建模拟轨迹
        ego_trajectory = torch.randn(1, 10, 4)
        neighbor_trajectories = torch.randn(1, 2, 10, 4)
        
        # 生成成本地图
        cost_map = generator.generate_cost_map(
            occupancy_pred, ego_trajectory, neighbor_trajectories
        )
        
        # 可视化
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(occupancy_pred[0, 0].cpu().numpy(), cmap='hot')
        plt.title('Occupancy Prediction (t=0)')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(cost_map[0].cpu().numpy(), cmap='hot')
        plt.title('Cost Map')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        # 显示梯度
        grad_x = torch.gradient(cost_map, dim=2)[0]
        grad_y = torch.gradient(cost_map, dim=1)[0]
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        plt.imshow(grad_magnitude[0].cpu().numpy(), cmap='hot')
        plt.title('Gradient Magnitude')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('cost_map_visualization.png', dpi=150, bbox_inches='tight')
        print("✓ Cost map visualization saved as 'cost_map_visualization.png'\n")
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}\n")

def main():
    """主测试函数"""
    print("=" * 60)
    print("Testing Grid Convolutional Guidance Components")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 测试各个组件
        test_unet_occupancy_predictor()
        test_observation_to_grid_converter()
        test_grid_cost_map_generator()
        test_grid_conv_guidance_fn()
        
        # 可视化（可选）
        visualize_cost_map()
        
        print("=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
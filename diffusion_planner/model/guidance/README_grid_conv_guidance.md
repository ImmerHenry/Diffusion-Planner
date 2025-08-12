# Grid Convolution Guidance for Diffusion Planner

## 概述

`grid_conv_guidance.py` 实现了基于UNET生成的栅格化成本地图的guidance函数，用于引导diffusion planner生成的轨迹避开高成本区域，前往低成本区域。

## 主要特性

- **UNET占用预测模型**: 基于21帧历史栅格数据预测30帧未来占用概率
- **栅格配置**: 400×400像素，0.25米/像素分辨率，覆盖100×100米区域
- **Ego-centric视角**: 与diffusion planner默认设置一致的自车中心坐标系
- **轨迹采样**: 在栅格成本图上对去噪轨迹进行采样
- **梯度引导**: 计算成本梯度，引导轨迹向低成本区域移动

## 核心组件

### 1. UNetOccupancyPredictor
```python
class UNetOccupancyPredictor(nn.Module):
    """
    UNET模型用于占用预测
    输入: [B, 21, 400, 400] - 21帧历史栅格数据
    输出: [B, 30, 400, 400] - 30帧未来占用概率预测
    """
```

### 2. 坐标变换函数
```python
def world_to_grid_coords(world_coords: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
    """
    将世界坐标转换为ego-centric栅格坐标
    
    Args:
        world_coords: [B, N, 2] 世界坐标 (x, y)
        ego_pose: [B, 3] ego位姿 (x, y, heading)
    
    Returns:
        grid_coords: [B, N, 2] 栅格坐标 (i, j)
    """
```

### 3. 轨迹采样函数
```python
def sample_trajectory_on_grid(trajectory: torch.Tensor, ego_pose: torch.Tensor, 
                             cost_map: torch.Tensor) -> torch.Tensor:
    """
    在成本栅格上采样轨迹点并计算成本
    
    使用PyTorch的F.grid_sample实现高效采样:
    - 自动处理边界情况 (padding_mode='border')
    - GPU优化的双线性插值
    - 支持梯度反向传播
    
    Args:
        trajectory: [B, T, 2] 世界坐标系下的轨迹
        ego_pose: [B, 3] 当前ego位姿
        cost_map: [B, FUTURE_FRAMES, GRID_SIZE, GRID_SIZE] 未来帧成本图
    
    Returns:
        costs: [B, T] 轨迹上各点的成本值
    """
```

### 4. 主要guidance函数
```python
def grid_conv_guidance_fn(x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, 
                         inputs: Dict, *args, **kwargs) -> torch.Tensor:
    """
    栅格卷积guidance函数，基于UNET生成的占用预测引导轨迹
    
    Args:
        x: [B, P, T*4] 轨迹张量 (B: batch, P: agents, T: time steps)
        t: [B] diffusion时间步
        cond: 条件信息
        inputs: 输入数据字典
        
    Returns:
        reward: [B] 每个batch的guidance奖励
    """
```

## 配置参数

```python
GRID_SIZE = 400          # 栅格尺寸 (400×400)
GRID_RESOLUTION = 0.25   # 栅格分辨率 (0.25米/像素)
GRID_RANGE = 100         # 栅格覆盖范围 (100米)
HISTORY_FRAMES = 21      # 历史帧数
FUTURE_FRAMES = 30       # 预测帧数
```

## 使用方法

### 1. 集成到guidance wrapper

在 `guidance_wrapper.py` 中已自动集成:

```python
from diffusion_planner.model.guidance.grid_conv_guidance import grid_conv_guidance_fn

class GuidanceWrapper:
    def __init__(self):
        self._guidance_fns = [
            collision_guidance_fn,
            grid_conv_guidance_fn
        ]
```

### 2. 配置diffusion planner

使用包含guidance的配置文件:

```bash
# 运行带有guidance的simulation
./sim_guidance_demo.sh
```

或在代码中设置:

```python
from diffusion_planner.model.guidance.guidance_wrapper import GuidanceWrapper

# 创建guidance wrapper
guidance_fn = GuidanceWrapper()

# 在decoder中使用
decoder_config.guidance_fn = guidance_fn
```

## 工作原理

1. **历史数据处理**: 接收21帧历史栅格数据 (可以是真实传感器数据或模拟数据)

2. **未来预测**: UNET模型基于历史数据预测30帧未来的占用概率

3. **成本图生成**: 
   - 占用概率直接作为成本
   - 添加膨胀处理，增加障碍物周围的成本

4. **轨迹采样**: 
   - 将轨迹从世界坐标转换到ego-centric栅格坐标
   - 使用PyTorch的`F.grid_sample`进行高效的双线性插值采样

5. **reward计算**:
   - 对轨迹成本进行时间加权 (近期时间步权重更高)
   - 转换为负reward (低成本 = 高reward)
   - 添加高成本区域的惩罚项

6. **梯度引导**: 通过自动微分计算梯度，引导diffusion过程

## 自定义和扩展

### 替换UNET模型

要使用训练好的UNET模型，替换 `create_mock_unet_predictor` 函数:

```python
def create_trained_unet_predictor(device: torch.device, model_path: str) -> UNetOccupancyPredictor:
    """加载训练好的UNET模型"""
    model = UNetOccupancyPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
```

### 调整成本函数

可以修改成本计算逻辑:

```python
# 在grid_conv_guidance_fn中修改
cost_map = future_occupancy.clone()

# 自定义成本函数
cost_map = custom_cost_function(future_occupancy, additional_features)
```

### 修改guidance强度

调整最终的reward缩放因子:

```python
return 2.0 * reward  # 增加guidance强度
return 0.5 * reward  # 减少guidance强度
```

## 注意事项

1. **坐标系一致性**: 确保输入的轨迹和UNET预测使用相同的ego-centric坐标系

2. **时间同步**: 轨迹时间步长应与UNET预测的时间步长匹配

3. **计算效率**: UNET推理会增加计算开销，可考虑:
   - 使用轻量级模型
   - 缓存预测结果
   - 降低更新频率

4. **guidance平衡**: 与collision guidance一起使用时，注意调整相对权重

## 性能优化建议

1. **模型优化**: 使用TensorRT或ONNX优化UNET模型
2. **批处理**: 批量处理多个轨迹的预测
3. **内存管理**: 及时释放中间变量
4. **GPU加速**: 确保所有计算在GPU上进行
5. **高效采样**: 使用`F.grid_sample`进行GPU优化的双线性插值，相比手动实现更快且支持自动微分

## 故障排除

### 常见问题

1. **内存不足**: 减小batch size或使用gradient checkpointing
2. **坐标错误**: 检查world_to_grid_coords的变换逻辑
3. **NaN values**: 确保除零保护和数值稳定性
4. **收敛问题**: 调整guidance权重和学习率

### 调试建议

启用调试模式查看中间结果:

```python
# 在grid_conv_guidance_fn中添加
if debug_mode:
    print(f"Trajectory costs: {trajectory_costs}")
    print(f"Cost map stats: min={cost_map.min()}, max={cost_map.max()}")
```
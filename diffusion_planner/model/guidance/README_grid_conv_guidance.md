# Grid Convolutional Guidance

## 概述

`grid_conv_guidance` 是一个基于UNet生成的栅格化成本地图的轨迹引导系统，用于在diffusion planner中引导轨迹去往低成本区域。

## 主要特性

- **UNet Occupancy Prediction**: 使用UNet模型从21帧历史观测预测30帧未来occupancy
- **栅格化成本地图**: 400x400网格，0.25m/pixel分辨率
- **梯度引导**: 计算成本地图梯度，引导轨迹去往低成本区域
- **ego-centric**: 与diffusion planner默认设置一致，以ego车辆为中心

## 架构组成

### 1. UNetOccupancyPredictor

UNet模型，用于生成occupancy prediction：

- **输入**: `[B, 21, C, H, W]` - 21帧历史观测
- **输出**: `[B, 30, H, W]` - 30帧未来occupancy概率
- **结构**: 4层encoder-decoder架构，带skip connections

### 2. ObservationToGridConverter

将观测数据转换为栅格化表示：

- 将智能体历史轨迹转换为栅格
- 处理ego车辆和邻居智能体
- 支持地图特征转换

### 3. GridCostMapGenerator

生成栅格化成本地图：

- 基于occupancy prediction生成基础成本
- 添加轨迹相关成本（ego轨迹奖励，邻居轨迹惩罚）
- 添加边界成本
- 使用高斯滤波平滑成本地图

## 使用方法

### 1. 配置文件

使用 `diffusion_planner_grid_conv_guidance.yaml` 配置文件：

```yaml
diffusion_planner:
  config:
    guidance_fn:
      _target_: diffusion_planner.model.guidance.guidance_wrapper.GuidanceWrapper

  past_trajectory_sampling:
    num_poses: 21  # 21帧历史
    time_horizon: 2.1

  future_trajectory_sampling:
    num_poses: 30  # 30帧未来
    time_horizon: 3.0

  grid_cost_map:
    grid_size: 400  # 400x400网格
    resolution: 0.25  # 0.25m/pixel
    cost_scale: 10.0  # 成本缩放因子
```

### 2. 集成到GuidanceWrapper

新的guidance已经自动集成到 `GuidanceWrapper` 中：

```python
from diffusion_planner.model.guidance.guidance_wrapper import GuidanceWrapper

# 自动包含 collision_guidance_fn 和 grid_conv_guidance_fn
guidance_wrapper = GuidanceWrapper()
```

### 3. 运行

使用配置运行diffusion planner：

```bash
python -m diffusion_planner.planner.planner \
  --config diffusion_planner/config/planner/diffusion_planner_grid_conv_guidance.yaml
```

## 工作原理

### 1. 观测到栅格转换

1. 从 `inputs` 中提取智能体历史轨迹
2. 将轨迹坐标转换为400x400网格索引
3. 在智能体周围创建占用区域

### 2. UNet Occupancy Prediction

1. 将栅格化观测输入UNet
2. 生成30帧未来occupancy预测
3. 输出0-1概率值

### 3. 成本地图生成

1. 基于occupancy生成基础成本
2. 添加ego轨迹奖励（负成本）
3. 添加邻居轨迹惩罚（正成本）
4. 添加边界成本
5. 使用高斯滤波平滑

### 4. 梯度引导

1. 计算成本地图的x和y方向梯度
2. 在轨迹位置采样梯度值
3. 计算梯度引导力
4. 返回负梯度方向的guidance能量

## 参数调优

### 成本地图参数

- `cost_scale`: 控制occupancy成本强度（默认10.0）
- `boundary_width`: 边界成本宽度（默认20像素）
- 轨迹成本范围：ego轨迹奖励(-1.0)，邻居轨迹惩罚(+2.0)

### 平滑参数

- `kernel_size`: 高斯滤波核大小（默认5）
- `sigma`: 高斯滤波标准差（默认1.0）

### 时间控制

- 只在 `0.005 < t < 0.1` 的时间步应用guidance
- 避免在扩散过程早期和晚期过度干扰

## 扩展功能

### 1. 地图集成

可以扩展 `convert_map_to_grid` 方法，集成实际的地图特征：

```python
def convert_map_to_grid(self, map_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 处理车道线、边界、交通标志等地图特征
    # 返回栅格化地图表示
    pass
```

### 2. 动态成本调整

可以根据场景动态调整成本参数：

```python
def adjust_costs_by_scenario(self, scenario_type: str):
    if scenario_type == "highway":
        self.cost_scale = 15.0  # 高速公路更高成本
    elif scenario_type == "urban":
        self.cost_scale = 8.0   # 城市道路较低成本
```

### 3. 多模态预测

可以扩展UNet支持多模态输出：

```python
class MultiModalUNet(nn.Module):
    def forward(self, x):
        # 输出多个可能的occupancy预测
        # 用于不确定性建模
        pass
```

## 性能优化

### 1. 批处理

- 支持批量处理多个轨迹
- 使用向量化操作减少循环

### 2. 内存管理

- 使用 `torch.no_grad()` 避免不必要的梯度计算
- 及时释放中间变量

### 3. GPU加速

- 所有计算都在GPU上进行
- 使用PyTorch原生操作最大化性能

## 故障排除

### 常见问题

1. **索引越界**: 确保轨迹坐标在网格范围内
2. **内存不足**: 减少batch size或网格分辨率
3. **梯度爆炸**: 调整cost_scale参数

### 调试技巧

1. 可视化成本地图和轨迹
2. 检查梯度值范围
3. 监控guidance能量变化

## 参考文献

- UNet架构: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Diffusion Models: "Denoising Diffusion Probabilistic Models"
- Cost Map Planning: "A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles"
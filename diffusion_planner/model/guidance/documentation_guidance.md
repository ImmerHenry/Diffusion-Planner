# Classifer Guidance Tutorial

## Available Guidance Functions

### 1. Collision Guidance (`collision.py`)
- Prevents collisions with other agents
- Uses vehicle bounding boxes and signed distance calculations
- Applied during specific diffusion timesteps

### 2. Grid Convolution Guidance (`grid_conv_guidance.py`)
- **NEW**: UNET-based occupancy prediction guidance
- Grid size: 400×400, resolution: 0.25m/pixel
- Uses 21 historical frames to predict 30 future frames
- Guides trajectories towards low-cost regions in ego-centric grid
- See `README_grid_conv_guidance.md` for detailed documentation

## Create your own guidance function

1. Create ``diffusion_planner/model/guidance/<my_guidance>.py``

```python
def my_guidance_fn(x, t, cond, inputs) -> torch.Tensor:
    """
    Your custom guidance function.
    
    Args:
        x: [B, P, T*4] trajectory tensor
        t: [B] diffusion timestep  
        cond: conditioning information
        inputs: dict containing input data
        
    Returns:
        reward: [B] guidance reward for each batch
    """
    # Your guidance logic here
    ...
    return reward
```

2. Add ``<my_guidance_fn>`` in ``diffusion_planner/model/guidance/guidance_wrapper.py``

```python
# diffusion_planner/model/guidance/guidance_wrapper.py

from diffusion_planner.model.guidance.<my_guidance> import <my_guidance_fn>

...

class GuidanceWrapper:
    def __init__(self):
        self._guidance_fns = [
            collision_guidance_fn,
            grid_conv_guidance_fn,
            <my_guidance_fn>,  # Add your guidance here
        ]

    def __call__(...):
        ...

...
```

3. Run ``sim_guidance_demo.sh``
4. Enjoy.
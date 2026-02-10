# Robot IR

> **The vLLM of Robotics** — Intermediate Representation for Real-time Robot Policy Execution

Robot IR is a unified intermediate layer for expressing real-time robot policy execution graphs:

```
Sensor → World Model → Policy → Action → Control Loop
```

## Why Robot IR?

Standard ML IRs (ONNX, TorchScript, etc.) lack primitives for:

- **Time**: Control frequency, multi-rate execution
- **State**: KV cache, world model memory, belief state
- **Memory**: Streaming buffers, temporal reuse
- **Real-time**: Latency constraints, deadline enforcement

Robot IR fills this gap with first-class support for embodied AI deployment.

## Core Features

### Time-Aware Graph
Support for control frequency and multi-rate execution (vision @ 10Hz, policy @ 20Hz, control @ 100Hz).

### Memory-Aware Execution
First-class support for:
- KV cache reuse
- State persistence
- Streaming sensor buffers

### Multi-Agent/Multi-Model Native
Natural support for multi-model coordination and expert routing.

### Hardware-Lowering Friendly
Easy lowering to:
- TensorRT
- Triton
- CUDA graphs
- ROS2 executor

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from robot_ir import RobotModule, SensorNode, Modality, TensorShape
from robot_ir.integration import Pi0Mapping

# Create Pi0 policy IR using the mapping
mapping = Pi0Mapping(
    image_size=224,
    action_dim=7,
    num_cameras=2,
)
module = mapping.create_module("pi0_policy")

# Validate
errors = module.validate()
assert not errors, f"Validation failed: {errors}"

# Serialize
ir_dict = module.to_dict()
```

## Architecture

```
RobotModule
 ├── Sensors          # Multi-modal inputs
 ├── StateMemory      # KV cache, world model memory
 ├── ComputeGraph     # Neural blocks, fusion, control logic
 ├── ControlLoop      # Timing constraints
 ├── SchedulingHints  # Hardware optimization hints
 └── DeploymentConfig # Target platform
```

## Supported Policies

| Policy | Status | Notes |
|--------|--------|-------|
| Pi0 | ✅ | SigLIP + Flow Matching |
| Pi0.5 | ✅ | PaLiGemma VLM backbone |
| ACT | 🔄 | Coming soon |
| Diffusion | 🔄 | Coming soon |
| OpenVLA | 🔄 | Coming soon |

## Integration with robot_runtime

Robot IR is designed to work seamlessly with [robot_runtime](https://github.com/LiangSu8899/robot_runtime):

```python
from robot_ir.integration import RuntimeBridge

bridge = RuntimeBridge()
adapter = bridge.to_runtime_adapter(ir_module)

# Use with robot_runtime
adapter.warmup(sample_obs)
action, state, stats = adapter.step(obs, state)
```

## Lowering Pipeline

```
Robot IR
    ↓ (optimization passes)
Execution IR
    ↓ (backend lowering)
Hardware Graph (TensorRT / Triton / CUDA)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check robot_ir/

# Type check
mypy robot_ir/
```

## License

Apache 2.0

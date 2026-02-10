# Robot IR - Development Roadmap

## Phase 1: Core IR ✅ COMPLETE

### Completed ✅
- [x] Core type definitions (TensorShape, DataType)
- [x] RobotModule top-level container
- [x] SensorNode with multi-modal support
- [x] StateBuffer for memory management
- [x] ComputeGraph and node types
- [x] ControlLoop specification
- [x] SchedulingHints for Thor optimization
- [x] Pi0/Pi0.5 model mappings (with full compute graph)
- [x] Basic lowering pass infrastructure
- [x] RuntimeBridge for robot_runtime integration
- [x] JSON serialization/deserialization
- [x] Unit tests for core components

## Phase 2: Pi-Series Model Integration

### Robot IR ↔ Pi Model Mapping ✅
- [x] Complete Pi0 architecture mapping
  - [x] Vision encoder (SigLIP) node spec
  - [x] Proprio encoder node spec
  - [x] Multi-modal fusion node spec
  - [x] Action expert node spec
  - [x] Flow matching head spec
  - [x] Action chunk selector node spec
- [x] Pi0.5 with VLM backbone
  - [x] PaLiGemma encoder mapping
  - [x] Language token handling
  - [x] Language KV cache
- [ ] Pi0-FAST variant
  - [ ] Autoregressive action tokenizer

### Memory Optimization Mapping 🔄
- [x] Action chunk buffer management
- [x] Vision feature caching
- [ ] KV cache layout specification
  - [ ] Layer-wise cache allocation
  - [ ] Sliding window support

### Runtime Integration ✅
- [x] RuntimeBridge.create_runtime_configs()
- [x] extract_observation_spec()
- [x] extract_action_spec()
- [x] get_adapter_type()
- [x] validate_for_runtime()

## Phase 3: Thor Backend

### TensorRT Integration
- [ ] IR to TensorRT engine compilation
- [ ] FP8/INT8 quantization support
- [ ] CUDA graph capture
- [ ] Multi-stream execution

### Memory Optimization
- [ ] Memory-bound operation detection
- [ ] Kernel fusion patterns
- [ ] Memory layout optimization

### Real-time Execution
- [ ] Deadline enforcement
- [ ] Latency profiling
- [ ] Jitter analysis

## Phase 4: Advanced Features

### Multi-Rate Execution
- [ ] Rate conversion buffers
- [ ] Async sensor handling
- [ ] Pipeline scheduling

### World Model Support
- [ ] Latent dynamics modeling
- [ ] Action rollout planning
- [ ] Belief state management

### Safety & Constraints
- [ ] Safety filter nodes
- [ ] Action bounds enforcement
- [ ] Emergency stop handling

## Phase 5: Ecosystem Integration

### ROS2 Bridge
- [ ] IR to ROS2 node graph
- [ ] Sensor topic mapping
- [ ] Action publisher integration

### Simulation
- [ ] MuJoCo integration
- [ ] Isaac Sim bridge
- [ ] Gym environment wrapper

### Debugging & Visualization
- [ ] IR graph visualization
- [ ] Execution trace logging
- [ ] Performance dashboard

---

## Priority Matrix (Updated)

| Feature | Impact | Effort | Priority | Status |
|---------|--------|--------|----------|--------|
| Pi0 mapping complete | High | Medium | P0 | ✅ Done |
| Runtime bridge | High | Medium | P0 | ✅ Done |
| JSON serialization | Medium | Low | P0 | ✅ Done |
| KV cache optimization | High | Medium | P1 | 🔄 Partial |
| TensorRT backend | High | High | P1 | Pending |
| Multi-rate execution | Medium | High | P2 | Pending |
| ROS2 integration | Medium | Medium | P2 | Pending |
| World model support | Medium | High | P3 | Pending |

---

## Next Steps (Immediate)

1. **Implement TensorRT lowering pass**
   - Convert neural blocks to TRT engines
   - Handle precision policies
   - Integrate CUDA graphs

2. **KV cache optimization pass**
   - Analyze reuse opportunities
   - Plan cache allocation
   - Generate eviction schedule

3. **End-to-end integration test**
   - Load real Pi0 policy
   - Create IR from policy
   - Generate runtime config
   - Execute with robot_runtime

4. **Add ACT/Diffusion policy mappings**
   - ACT architecture mapping
   - Diffusion policy mapping
   - OpenVLA mapping

---

## Changelog

### v0.1.1 (Current)
- Added complete Pi0 compute graph mapping
- Implemented RuntimeBridge with full config generation
- Added JSON serialization/deserialization
- Added integration and serialization tests
- Updated Pi0Config to match robot_runtime structure

### v0.1.0
- Initial release with core IR types
- Basic Pi0/Pi0.5 mapping
- Lowering pass infrastructure

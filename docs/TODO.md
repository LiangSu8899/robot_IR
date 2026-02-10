# Robot IR - Development Roadmap

## Phase 1: Core IR (Current)

### Completed ✅
- [x] Core type definitions (TensorShape, DataType)
- [x] RobotModule top-level container
- [x] SensorNode with multi-modal support
- [x] StateBuffer for memory management
- [x] ComputeGraph and node types
- [x] ControlLoop specification
- [x] SchedulingHints for Thor optimization
- [x] Pi0/Pi0.5 model mappings
- [x] Basic lowering pass infrastructure

### In Progress 🔄
- [ ] Complete lowering pass implementations
- [ ] Runtime bridge to robot_runtime
- [ ] Serialization/deserialization (JSON, Protobuf)

## Phase 2: Pi-Series Model Integration

### Robot IR ↔ Pi Model Mapping
- [ ] Complete Pi0 architecture mapping
  - [ ] Vision encoder (SigLIP) node spec
  - [ ] Action expert node spec
  - [ ] Flow matching head spec
- [ ] Pi0.5 with VLM backbone
  - [ ] PaLiGemma encoder mapping
  - [ ] Language token handling
- [ ] Pi0-FAST variant
  - [ ] Autoregressive action tokenizer

### Memory Optimization Mapping
- [ ] KV cache layout specification
  - [ ] Layer-wise cache allocation
  - [ ] Sliding window support
- [ ] Action chunk buffer management
- [ ] Vision feature caching

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

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Pi0 mapping complete | High | Medium | P0 |
| KV cache optimization | High | Medium | P0 |
| TensorRT backend | High | High | P0 |
| Runtime bridge | High | Medium | P1 |
| Multi-rate execution | Medium | High | P1 |
| ROS2 integration | Medium | Medium | P2 |
| World model support | Medium | High | P2 |

---

## Next Steps (Immediate)

1. **Complete Pi0 IR mapping validation**
   - Verify compute graph matches model structure
   - Test serialization/deserialization

2. **Implement RuntimeBridge.to_runtime_adapter()**
   - Map IR sensors to Observation
   - Map IR graph to policy model
   - Configure runtime from IR

3. **Add TensorRT lowering pass**
   - Convert neural blocks to TRT engines
   - Handle precision policies
   - Integrate CUDA graphs

4. **KV cache optimization pass**
   - Analyze reuse opportunities
   - Plan cache allocation
   - Generate eviction schedule

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

## Phase 2: Policy Model Integration ✅

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

### ACT Mapping ✅ (NEW)
- [x] ACT architecture mapping
  - [x] Vision backbone (ResNet18)
  - [x] Transformer encoder/decoder
  - [x] CVAE latent sampling
  - [x] Temporal ensemble for smoothing
- [x] ACTConfig with full parameter support
- [x] 50Hz control frequency support

### Diffusion Policy Mapping ✅ (NEW)
- [x] Diffusion Policy architecture mapping
  - [x] U-Net denoiser option
  - [x] Transformer denoiser option
  - [x] DDIM sampling loop
  - [x] Action trajectory prediction
- [x] DiffusionPolicyConfig with scheduler selection
- [x] Multi-step inference support

### OpenVLA Mapping ✅ (NEW)
- [x] OpenVLA architecture mapping
  - [x] Vision encoder + projector
  - [x] LLM backbone integration
  - [x] Autoregressive action generation
  - [x] Action tokenizer/detokenizer
- [x] OpenVLAConfig with VLM settings
- [x] Language instruction support
- [x] LLM KV cache management

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
- [x] get_policy_mapping() utility
- [x] list_supported_policies() utility

## Phase 3: Thor Backend

### TensorRT Integration ✅
- [x] IR to TensorRT engine compilation (TRTEngineBuilder)
- [x] FP8/INT8 quantization support (TRTPrecision mapping)
- [x] CUDA graph capture (TRTEngineRunner)
- [x] Multi-stream execution (TRTExecutionPlanner)

### Memory Optimization 🔄
- [x] KV cache analysis and optimization
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
| KV cache optimization | High | Medium | P1 | ✅ Done |
| TensorRT backend | High | High | P1 | ✅ Done |
| ACT/Diffusion/OpenVLA mappings | High | Medium | P1 | ✅ Done |
| Multi-rate execution | Medium | High | P2 | Pending |
| ROS2 integration | Medium | Medium | P2 | Pending |
| World model support | Medium | High | P3 | Pending |

---

## Supported Policy Architectures

| Policy | Frequency | Action Type | Key Features |
|--------|-----------|-------------|--------------|
| Pi0 | 20 Hz | Chunk (50 steps) | Flow matching, vision-proprio fusion |
| Pi0.5 | 10 Hz | Chunk (50 steps) | VLM backbone, language conditioning |
| ACT | 50 Hz | Chunk (100 steps) | CVAE, temporal ensemble |
| Diffusion | 10 Hz | Trajectory (16 steps) | DDIM sampling, multi-modal |
| OpenVLA | 5 Hz | Tokenized (7 tokens) | LLM backbone, language instructions |

---

## Next Steps (Immediate)

1. **Multi-rate execution implementation**
   - Rate conversion buffers
   - Async sensor handling
   - Pipeline scheduling

2. **End-to-end integration with real models**
   - Load real Pi0/Pi0.5 policies
   - Run inference through IR pipeline
   - Benchmark against native execution

3. **ROS2 bridge implementation**
   - Sensor topic mapping
   - Action publisher integration
   - Executor graph generation

4. **Pi0-FAST variant support**
   - Autoregressive action tokenizer
   - Token-based action generation

---

## Changelog

### v0.1.3 (Current)
- Added ACT (Action Chunking with Transformers) mapping
- Added Diffusion Policy mapping with U-Net/Transformer options
- Added OpenVLA mapping with LLM backbone support
- Extended ControlLogicType with LOOP, GENERATOR, AGGREGATOR
- Extended PipelineMode with STREAMING, STANDARD
- Added NeuralBlock metadata field
- Added get_policy_mapping() and list_supported_policies() utilities
- Policy comparison example (03_policy_comparison.py)
- 26 new unit tests for policy mappings

### v0.1.2
- TensorRT lowering pass with precision mapping
- TRT engine builder and runner infrastructure
- KV cache analyzer and optimizer
- Sliding window, paged attention, flash attention support
- Runtime integration example
- Execution planner for stream assignment

### v0.1.1
- Added complete Pi0 compute graph mapping
- Implemented RuntimeBridge with full config generation
- Added JSON serialization/deserialization
- Added integration and serialization tests
- Updated Pi0Config to match robot_runtime structure

### v0.1.0
- Initial release with core IR types
- Basic Pi0/Pi0.5 mapping
- Lowering pass infrastructure

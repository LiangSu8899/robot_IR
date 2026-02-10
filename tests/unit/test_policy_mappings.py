"""Unit tests for ACT, Diffusion Policy, and OpenVLA mappings."""

import sys
sys.path.insert(0, ".")

import pytest

from robot_ir.integration import (
    ACTMapping,
    ACTConfig,
    DiffusionPolicyMapping,
    DiffusionPolicyConfig,
    DiffusionScheduler,
    OpenVLAMapping,
    OpenVLAConfig,
    get_policy_mapping,
    list_supported_policies,
)
from robot_ir.core.module import RobotModule
from robot_ir.compute.nodes import NeuralBlock, ControlLogicNode
from robot_ir.memory.state import ReuseStrategy


class TestACTMapping:
    """Tests for ACT policy mapping."""

    def test_create_module(self):
        """Test creating ACT module."""
        mapping = ACTMapping()
        module = mapping.create_module("test_act")

        assert module.name == "test_act"
        assert isinstance(module, RobotModule)

    def test_sensors(self):
        """Test ACT sensors configuration."""
        config = ACTConfig(num_cameras=2, camera_names=["cam_high", "cam_wrist"])
        mapping = ACTMapping(config=config)
        module = mapping.create_module()

        # Should have 2 cameras + proprio
        assert len(module.sensors) == 3

        # Check camera sensors
        cam_sensors = [s for s in module.sensors if "cam" in s.name]
        assert len(cam_sensors) == 2

        # Check proprio sensor
        proprio = module.get_sensor("qpos")
        assert proprio is not None

    def test_state_buffers(self):
        """Test ACT state buffers."""
        config = ACTConfig(use_temporal_ensemble=True)
        mapping = ACTMapping(config=config)
        module = mapping.create_module()

        # Should have action chunk, temporal ensemble, and cvae latent
        assert len(module.states) >= 3

        action_buffer = module.get_state("action_chunk")
        assert action_buffer is not None
        assert action_buffer.reuse_strategy == ReuseStrategy.ACTION_BUFFER

        ensemble = module.get_state("temporal_ensemble")
        assert ensemble is not None

    def test_compute_graph(self):
        """Test ACT compute graph structure."""
        mapping = ACTMapping()
        module = mapping.create_module()

        assert module.graph is not None

        # Check key nodes exist
        node_names = [n.name for n in module.graph.nodes]
        assert "vision_backbone" in node_names
        assert "transformer_encoder" in node_names
        assert "transformer_decoder" in node_names
        assert "cvae_sampler" in node_names
        assert "action_head" in node_names

        # Check entry/exit points
        assert "vision_backbone" in module.graph.entry_points
        assert len(module.graph.exit_points) > 0

    def test_control_loop(self):
        """Test ACT control loop configuration."""
        config = ACTConfig(control_frequency_hz=50.0, latency_budget_ms=20.0)
        mapping = ACTMapping(config=config)
        module = mapping.create_module()

        assert module.control is not None
        assert module.control.frequency_hz == 50.0
        assert module.control.latency_budget_ms == 20.0

    def test_runtime_config(self):
        """Test ACT runtime config generation."""
        mapping = ACTMapping()
        config = mapping.to_runtime_config()

        assert "deadline_ms" in config
        assert "backend" in config
        assert config["backend"] == "cuda_graph"


class TestDiffusionPolicyMapping:
    """Tests for Diffusion Policy mapping."""

    def test_create_module(self):
        """Test creating Diffusion Policy module."""
        mapping = DiffusionPolicyMapping()
        module = mapping.create_module("test_diffusion")

        assert module.name == "test_diffusion"
        assert isinstance(module, RobotModule)

    def test_diffusion_config(self):
        """Test diffusion-specific configuration."""
        config = DiffusionPolicyConfig(
            num_diffusion_steps=100,
            num_inference_steps=10,
            scheduler=DiffusionScheduler.DDIM,
        )
        mapping = DiffusionPolicyMapping(config=config)
        module = mapping.create_module()

        # Check denoiser node metadata
        denoiser = module.graph.get_node("denoiser")
        assert denoiser is not None
        assert isinstance(denoiser, NeuralBlock)
        assert denoiser.metadata["num_inference_steps"] == 10
        assert denoiser.metadata["scheduler"] == "ddim"

    def test_unet_vs_transformer(self):
        """Test U-Net vs Transformer denoiser."""
        # U-Net (default)
        unet_config = DiffusionPolicyConfig(use_transformer=False)
        unet_module = DiffusionPolicyMapping(config=unet_config).create_module()
        unet_denoiser = unet_module.graph.get_node("denoiser")
        assert unet_denoiser.model_ref == "diffusion_unet"

        # Transformer
        trans_config = DiffusionPolicyConfig(use_transformer=True)
        trans_module = DiffusionPolicyMapping(config=trans_config).create_module()
        trans_denoiser = trans_module.graph.get_node("denoiser")
        assert trans_denoiser.model_ref == "diffusion_transformer"

    def test_state_buffers(self):
        """Test Diffusion Policy state buffers."""
        mapping = DiffusionPolicyMapping()
        module = mapping.create_module()

        # Should have action trajectory and noise buffer
        action_traj = module.get_state("action_trajectory")
        assert action_traj is not None

        noise_buf = module.get_state("noise_trajectory")
        assert noise_buf is not None

    def test_diffusion_loop_node(self):
        """Test diffusion loop control node."""
        config = DiffusionPolicyConfig(num_inference_steps=20)
        mapping = DiffusionPolicyMapping(config=config)
        module = mapping.create_module()

        loop_node = module.graph.get_node("diffusion_loop")
        assert loop_node is not None
        assert isinstance(loop_node, ControlLogicNode)
        assert loop_node.config["num_iterations"] == 20

    def test_scheduling_no_cuda_graph(self):
        """Test that diffusion disables CUDA graphs (dynamic loop)."""
        mapping = DiffusionPolicyMapping()
        module = mapping.create_module()

        assert module.schedule is not None
        assert module.schedule.enable_cuda_graphs is False


class TestOpenVLAMapping:
    """Tests for OpenVLA mapping."""

    def test_create_module(self):
        """Test creating OpenVLA module."""
        mapping = OpenVLAMapping()
        module = mapping.create_module("test_openvla")

        assert module.name == "test_openvla"
        assert isinstance(module, RobotModule)

    def test_language_sensor(self):
        """Test OpenVLA language sensor."""
        config = OpenVLAConfig(max_language_tokens=512)
        mapping = OpenVLAMapping(config=config)
        module = mapping.create_module()

        lang_sensor = module.get_sensor("language_instruction")
        assert lang_sensor is not None
        assert lang_sensor.shape.dims == (512,)

    def test_kv_cache(self):
        """Test OpenVLA LLM KV cache."""
        mapping = OpenVLAMapping()
        module = mapping.create_module()

        kv_cache = module.get_state("llm_kv_cache")
        assert kv_cache is not None
        assert kv_cache.reuse_strategy == ReuseStrategy.KV_CACHE

    def test_compute_graph_structure(self):
        """Test OpenVLA compute graph."""
        mapping = OpenVLAMapping()
        module = mapping.create_module()

        node_names = [n.name for n in module.graph.nodes]

        # Check VLM components
        assert "vision_encoder" in node_names
        assert "vision_projector" in node_names
        assert "language_embedding" in node_names
        assert "llm_backbone" in node_names
        assert "action_generator" in node_names
        assert "action_detokenizer" in node_names

    def test_autoregressive_action(self):
        """Test autoregressive action generation node."""
        config = OpenVLAConfig(action_dim=7, action_vocab_size=256)
        mapping = OpenVLAMapping(config=config)
        module = mapping.create_module()

        action_gen = module.graph.get_node("action_generator")
        assert action_gen is not None
        assert isinstance(action_gen, ControlLogicNode)
        assert action_gen.config["mode"] == "autoregressive"
        assert action_gen.config["num_tokens"] == 7

    def test_slow_frequency(self):
        """Test OpenVLA has slower control frequency due to LLM."""
        config = OpenVLAConfig(control_frequency_hz=5.0)
        mapping = OpenVLAMapping(config=config)
        module = mapping.create_module()

        assert module.control.frequency_hz == 5.0

    def test_kv_reuse_high_threshold(self):
        """Test OpenVLA has high KV reuse threshold."""
        mapping = OpenVLAMapping()
        module = mapping.create_module()

        assert module.schedule.kv_reuse_threshold >= 0.9


class TestGetPolicyMapping:
    """Tests for get_policy_mapping utility."""

    def test_list_supported_policies(self):
        """Test listing supported policies."""
        policies = list_supported_policies()

        assert "pi0" in policies
        assert "pi0.5" in policies
        assert "act" in policies
        assert "diffusion_policy" in policies
        assert "openvla" in policies

    def test_get_policy_by_name(self):
        """Test getting policies by name."""
        act = get_policy_mapping("act")
        assert isinstance(act, ACTMapping)

        diff = get_policy_mapping("diffusion")
        assert isinstance(diff, DiffusionPolicyMapping)

        vla = get_policy_mapping("openvla")
        assert isinstance(vla, OpenVLAMapping)

    def test_case_insensitive(self):
        """Test case-insensitive policy lookup."""
        act1 = get_policy_mapping("ACT")
        act2 = get_policy_mapping("act")
        act3 = get_policy_mapping("Act")

        assert type(act1) == type(act2) == type(act3)

    def test_unknown_policy_error(self):
        """Test error on unknown policy."""
        with pytest.raises(ValueError) as exc_info:
            get_policy_mapping("unknown_policy")

        assert "Unknown policy" in str(exc_info.value)


class TestPolicyComparison:
    """Tests comparing different policy architectures."""

    def test_all_policies_create_valid_modules(self):
        """Test that all policies create valid modules."""
        for policy_name in list_supported_policies():
            mapping = get_policy_mapping(policy_name)
            module = mapping.create_module(f"{policy_name}_test")

            assert module is not None
            assert module.graph is not None
            assert len(module.graph.nodes) > 0
            assert len(module.graph.entry_points) > 0
            assert len(module.graph.exit_points) > 0

    def test_all_policies_have_runtime_config(self):
        """Test that all policies generate runtime configs."""
        for policy_name in list_supported_policies():
            mapping = get_policy_mapping(policy_name)
            config = mapping.to_runtime_config()

            assert isinstance(config, dict)
            assert "deadline_ms" in config
            assert "device" in config

    def test_frequency_varies_by_policy(self):
        """Test that different policies have appropriate frequencies."""
        frequencies = {}

        for policy_name in list_supported_policies():
            mapping = get_policy_mapping(policy_name)
            module = mapping.create_module()
            frequencies[policy_name] = module.control.frequency_hz

        # ACT should be fastest (50 Hz)
        assert frequencies.get("act", 0) >= 50.0

        # OpenVLA should be slowest (due to LLM)
        assert frequencies.get("openvla", 100) <= 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

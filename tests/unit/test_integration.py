"""Tests for integration module."""

import pytest

from robot_ir.integration.pi_mapping import Pi0Mapping, Pi0Config, get_policy_mapping
from robot_ir.integration.runtime_bridge import RuntimeBridge


class TestPi0Mapping:
    """Tests for Pi0 mapping."""

    def test_create_module(self):
        mapping = Pi0Mapping()
        module = mapping.create_module("test_pi0")

        assert module.name == "test_pi0"
        assert len(module.sensors) == 3  # 2 cameras + proprio
        assert len(module.states) == 3  # action_chunk, chunk_index, vision_cache
        assert module.graph is not None
        assert module.control is not None
        assert module.schedule is not None

    def test_custom_config(self):
        config = Pi0Config(
            image_size=256,
            action_dim=14,
            num_cameras=1,
            chunk_size=100,
        )
        mapping = Pi0Mapping(config=config)
        module = mapping.create_module()

        # Check action chunk shape
        action_state = module.get_state("action_chunk")
        assert action_state is not None
        assert action_state.shape.dims == (100, 14)

        # Check sensor count
        vision_sensors = [s for s in module.sensors if s.modality.name == "VISION"]
        assert len(vision_sensors) == 1

    def test_compute_graph_structure(self):
        mapping = Pi0Mapping()
        module = mapping.create_module()

        graph = module.graph
        assert graph is not None

        # Check nodes
        assert graph.get_node("vision_encoder") is not None
        assert graph.get_node("proprio_encoder") is not None
        assert graph.get_node("multimodal_fusion") is not None
        assert graph.get_node("action_expert") is not None
        assert graph.get_node("flow_matching_head") is not None
        assert graph.get_node("action_chunk_selector") is not None

        # Check entry/exit points
        assert "vision_encoder" in graph.entry_points
        assert "action_chunk_selector" in graph.exit_points

    def test_to_runtime_config(self):
        mapping = Pi0Mapping()
        config = mapping.to_runtime_config()

        assert "deadline_ms" in config
        assert "backend" in config
        assert config["backend"] == "cuda_graph"

    def test_to_pi0_config(self):
        mapping = Pi0Mapping()
        config = mapping.to_pi0_config()

        assert "chunk_size" in config
        assert "temporal_action_steps" in config
        assert "use_action_cache" in config


class TestRuntimeBridge:
    """Tests for runtime bridge."""

    def test_create_runtime_configs(self):
        mapping = Pi0Mapping()
        module = mapping.create_module()

        bridge = RuntimeBridge()
        runtime_config, adapter_config = bridge.create_runtime_configs(module)

        assert "deadline_ms" in runtime_config
        assert "backend" in runtime_config
        assert "chunk_size" in adapter_config

    def test_extract_observation_spec(self):
        mapping = Pi0Mapping()
        module = mapping.create_module()

        bridge = RuntimeBridge()
        spec = bridge.extract_observation_spec(module)

        assert "images" in spec
        assert "state_dim" in spec
        assert len(spec["images"]) == 2  # 2 cameras
        assert spec["state_dim"] == 14

    def test_extract_action_spec(self):
        mapping = Pi0Mapping()
        module = mapping.create_module()

        bridge = RuntimeBridge()
        spec = bridge.extract_action_spec(module)

        assert spec["chunk_size"] == 50
        assert spec["action_dim"] == 7

    def test_get_adapter_type(self):
        bridge = RuntimeBridge()

        pi0_mapping = Pi0Mapping()
        pi0_module = pi0_mapping.create_module("my_pi0_policy")
        assert bridge.get_adapter_type(pi0_module) == "pi0"

    def test_validate_for_runtime(self):
        mapping = Pi0Mapping()
        module = mapping.create_module()

        bridge = RuntimeBridge()
        errors = bridge.validate_for_runtime(module)

        assert len(errors) == 0  # Should be valid


class TestPolicyMapping:
    """Tests for policy mapping factory."""

    def test_get_pi0_mapping(self):
        mapping = get_policy_mapping("pi0")
        assert isinstance(mapping, Pi0Mapping)

    def test_get_pi05_mapping(self):
        mapping = get_policy_mapping("pi0.5")
        from robot_ir.integration.pi_mapping import Pi05Mapping
        assert isinstance(mapping, Pi05Mapping)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            get_policy_mapping("unknown_policy")

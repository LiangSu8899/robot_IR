"""Tests for serialization module."""

import json
import pytest

from robot_ir.core.serialization import IRSerializer, IRDeserializer, to_json, from_json
from robot_ir.integration.pi_mapping import Pi0Mapping


class TestSerialization:
    """Tests for IR serialization."""

    def test_serialize_to_json(self):
        mapping = Pi0Mapping()
        module = mapping.create_module("test_module")

        serializer = IRSerializer()
        json_str = serializer.to_json(module)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "version" in data
        assert "module" in data
        assert data["module"]["name"] == "test_module"

    def test_serialize_to_dict(self):
        mapping = Pi0Mapping()
        module = mapping.create_module()

        serializer = IRSerializer()
        data = serializer.to_dict(module)

        assert data["module"]["name"] == "pi0_policy"
        assert len(data["module"]["sensors"]) == 3
        assert len(data["module"]["states"]) == 3

    def test_deserialize_from_json(self):
        mapping = Pi0Mapping()
        original = mapping.create_module("round_trip_test")

        # Serialize
        json_str = to_json(original)

        # Deserialize
        restored = from_json(json_str)

        # Verify
        assert restored.name == original.name
        assert len(restored.sensors) == len(original.sensors)
        assert len(restored.states) == len(original.states)

    def test_round_trip_sensors(self):
        mapping = Pi0Mapping()
        original = mapping.create_module()

        json_str = to_json(original)
        restored = from_json(json_str)

        # Check sensors match
        for orig_sensor, rest_sensor in zip(original.sensors, restored.sensors):
            assert orig_sensor.name == rest_sensor.name
            assert orig_sensor.modality == rest_sensor.modality
            assert orig_sensor.shape.dims == rest_sensor.shape.dims
            assert orig_sensor.rate_hz == rest_sensor.rate_hz

    def test_round_trip_states(self):
        mapping = Pi0Mapping()
        original = mapping.create_module()

        json_str = to_json(original)
        restored = from_json(json_str)

        # Check states match
        for orig_state, rest_state in zip(original.states, restored.states):
            assert orig_state.name == rest_state.name
            assert orig_state.shape.dims == rest_state.shape.dims
            assert orig_state.persistence == rest_state.persistence
            assert orig_state.reuse_strategy == rest_state.reuse_strategy

    def test_round_trip_graph(self):
        mapping = Pi0Mapping()
        original = mapping.create_module()

        json_str = to_json(original)
        restored = from_json(json_str)

        assert restored.graph is not None
        assert len(restored.graph.nodes) == len(original.graph.nodes)
        assert len(restored.graph.edges) == len(original.graph.edges)

    def test_round_trip_control(self):
        mapping = Pi0Mapping()
        original = mapping.create_module()

        json_str = to_json(original)
        restored = from_json(json_str)

        assert restored.control is not None
        assert restored.control.frequency_hz == original.control.frequency_hz
        assert restored.control.latency_budget_ms == original.control.latency_budget_ms

    def test_round_trip_schedule(self):
        mapping = Pi0Mapping()
        original = mapping.create_module()

        json_str = to_json(original)
        restored = from_json(json_str)

        assert restored.schedule is not None
        assert restored.schedule.enable_cuda_graphs == original.schedule.enable_cuda_graphs
        assert restored.schedule.tensorrt_enabled == original.schedule.tensorrt_enabled

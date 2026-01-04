"""
Black box tests for StateManagementSubsystem.

Tests validate the contract specified in documentation/base_object_implementation.md.
All tests use black box methodology:
- Test only documented behavior
- Observe state through public methods
- Verify side effects on injected optimizer (param_groups extension)
- Never test implementation details

Test organization:
- get_state() retrieval and aggregation
- set_state() storage and flag enforcement
- show_state() enumeration and filtering
- state_dict()/load_state_dict() serialization
- ScheduleAnything integration (observable side effects)
"""

import pytest
import torch
import torch.nn as nn

from src.gradient_quality_control.base.state_management import StateManagementSubsystem


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


def create_simple_optimizer(lr=0.01, weight_decay=0.0001):
    """Create simple optimizer with single parameter group."""
    params = [nn.Parameter(torch.randn(10, 5))]
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)


def create_multigroup_optimizer(lr1=0.01, lr2=0.001, wd1=0.0001, wd2=0.0002):
    """Create optimizer with multiple parameter groups."""
    params1 = [nn.Parameter(torch.randn(10, 5))]
    params2 = [nn.Parameter(torch.randn(5, 2))]

    return torch.optim.SGD([
        {'params': params1, 'lr': lr1, 'weight_decay': wd1},
        {'params': params2, 'lr': lr2, 'weight_decay': wd2}
    ])


def create_uniform_multigroup_optimizer(lr=0.01, weight_decay=0.0001):
    """Create multigroup optimizer where all groups have same lr/wd."""
    params1 = [nn.Parameter(torch.randn(10, 5))]
    params2 = [nn.Parameter(torch.randn(5, 2))]

    return torch.optim.SGD([
        {'params': params1, 'lr': lr, 'weight_decay': weight_decay},
        {'params': params2, 'lr': lr, 'weight_decay': weight_decay}
    ])


# =============================================================================
# Get State Test Suite - tests that get_state() retrieves wrapper state values
# =============================================================================


class TestGetStateWrapperStates:
    """Test get_state() retrieval from wrapper_states."""

    def test_get_state_retrieves_vital_state(self):
        """get_state() retrieves values stored with vital flag."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("test_metric", 42.0, "vital")

        value = state_mgr.get_state("test_metric")
        assert value == 42.0

    def test_get_state_retrieves_optional_state(self):
        """get_state() retrieves values stored with optional flag."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("debug_info", "test_value", "optional")

        value = state_mgr.get_state("debug_info")
        assert value == "test_value"

    def test_get_state_returns_updated_value(self):
        """get_state() returns most recent value after updates."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("counter", 0, "vital")
        state_mgr.set_state("counter", 5, "vital")
        state_mgr.set_state("counter", 10, "vital")

        value = state_mgr.get_state("counter")
        assert value == 10

    def test_get_state_returns_reference_to_mutable_wrapper_state(self):
        """get_state() returns reference to mutable objects, not copies."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("my_list", [1, 2, 3], "vital")

        # Get reference and mutate it
        retrieved = state_mgr.get_state("my_list")
        retrieved.append(4)

        # Get again - should see mutation
        retrieved2 = state_mgr.get_state("my_list")
        assert retrieved2 == [1, 2, 3, 4]


# =============================================================================
# Get State Test Suite - tests that get_state() retrieves optimizer parameters as lists
# =============================================================================


class TestGetStateOptimizerParams:
    """Test get_state() retrieval from optimizer param_groups."""

    def test_get_state_default_returns_list_for_single_group(self):
        """get_state() with default aggregate_behavior returns list even for single group."""
        optimizer = create_simple_optimizer(lr=0.123)
        state_mgr = StateManagementSubsystem(optimizer)

        lr = state_mgr.get_state("lr")
        assert isinstance(lr, list)
        assert lr == [0.123]


    def test_get_state_default_returns_list_for_uniform_multigroup(self):
        """get_state() default returns list even when all groups have same value."""
        optimizer = create_uniform_multigroup_optimizer(lr=0.01)
        state_mgr = StateManagementSubsystem(optimizer)

        lr = state_mgr.get_state("lr")
        assert isinstance(lr, list)
        assert len(lr) == 2
        assert all(val == 0.01 for val in lr)

    def test_get_state_returns_scalar_tensor(self):
        """get_state() returns scalar tensor."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        optimizer.param_groups[0]["test_metric"] = torch.tensor(42.0)

        test_list = state_mgr.get_state("test_list")
        assert isinstance(test_list, list)
        assert len(test_list) == 1
        assert test_list[0] == torch.tensor(42.0)


# =============================================================================
# Get State Test Suite - tests that get_state() raises appropriate errors for invalid inputs
# =============================================================================


class TestGetStateErrors:
    """Test get_state() error conditions."""

    def test_get_state_throws_on_nonexistent_name(self):
        """get_state() throws error when name not found."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        with pytest.raises(KeyError):
            state_mgr.get_state("nonexistent_key")

    def test_get_state_throws_on_non_numeric_optimizer_param(self):
        """get_state() throws error for non-numeric optimizer params."""
        optimizer = create_simple_optimizer()
        # Manually add non-numeric param to test filtering
        optimizer.param_groups[0]['string_param'] = "not_a_number"

        state_mgr = StateManagementSubsystem(optimizer)

        with pytest.raises(TypeError):
            state_mgr.get_state("string_param")

    def test_get_state_throws_on_non_scalar_tensor(self):
        """get_state() should not be used directly to retrieve parameters"""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        with pytest.raises(TypeError):
            state_mgr.get_state("params")

# =============================================================================
# Set State Test Suite - tests that set_state() stores values with different flags
# =============================================================================

class TestSetStateBasic:
    """Test set_state() basic storage functionality."""

    def test_set_state_vital_stores_and_retrieves(self):
        """set_state() with flag='vital' stores value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("metric", 99.9, "vital")
        value = state_mgr.get_state("metric")

        assert value == 99.9

    def test_set_state_optional_stores_and_retrieves(self):
        """set_state() with flag='optional' stores value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("debug", "test", "optional")
        value = state_mgr.get_state("debug")

        assert value == "test"

    def test_set_state_allows_updating_value_with_same_flag(self):
        """set_state() allows updating value when flag matches."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("counter", 0, "vital")
        state_mgr.set_state("counter", 5, "vital")  # Should not throw

        value = state_mgr.get_state("counter")
        assert value == 5

    def test_set_state_stores_various_types(self):
        """set_state() stores various Python types."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("int_val", 42, "vital")
        state_mgr.set_state("float_val", 3.14, "vital")
        state_mgr.set_state("str_val", "hello", "optional")
        state_mgr.set_state("list_val", [1, 2, 3], "optional")
        state_mgr.set_state("none_val", None, "vital")

        assert state_mgr.get_state("int_val") == 42
        assert state_mgr.get_state("float_val") == 3.14
        assert state_mgr.get_state("str_val") == "hello"
        assert state_mgr.get_state("list_val") == [1, 2, 3]
        assert state_mgr.get_state("none_val") is None


# =============================================================================
# Set State Test Suite - tests that set_state() checks error conditions
# =============================================================================


class TestSetStateFlagEnforcement:
    """Test set_state() flag immutability and validation."""

    def test_set_state_throws_when_changing_flag(self):
        """set_state() throws error when changing flag for existing name."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("metric", 1.0, "vital")

        # Attempting to change flag should throw
        with pytest.raises(RuntimeError):
            state_mgr.set_state("metric", 2.0, "optional")

    def test_set_state_throws_when_vital_collides_with_optimizer_param(self):
        """set_state() throws when vital name collides with optimizer param."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        # 'lr' already exists in optimizer.param_groups
        with pytest.raises(RuntimeError):
            state_mgr.set_state("lr", 0.5, "vital")

    def test_set_state_throws_when_optional_collides_with_optimizer_param(self):
        """set_state() throws when optional name collides with optimizer param."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        with pytest.raises(RuntimeError):
            state_mgr.set_state("weight_decay", 0.01, "optional")


# =============================================================================
# Set State Test Suite - tests that set_state() extends optimizer param_groups
# =============================================================================


class TestSetStateOptimizerExtension:
    """Test set_state() with flag='optimizer' extends param_groups (observable side effect)."""

    def test_set_state_optimizer_extends_param_groups(self):
        """set_state() with flag='optimizer' adds param to all param_groups."""
        optimizer = create_multigroup_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        # Extend optimizer with custom parameter
        state_mgr.set_state("custom_threshold", 0.95, "optimizer")

        # Observable side effect: param_groups should now contain custom_threshold
        for group in optimizer.param_groups:
            assert "custom_threshold" in group
            assert group["custom_threshold"] == 0.95

    def test_set_state_optimizer_retrievable_via_get_state(self):
        """Extended optimizer params are retrievable via get_state()."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("learning_rate_multiplier", 2.0, "optimizer")

        value = state_mgr.get_state("learning_rate_multiplier")
        assert value == [2.0]

    def test_set_state_optimizer_all_groups(self):
        """Updating optimizer param via set_state updates all param_groups."""
        optimizer = create_multigroup_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("threshold", 1.0, "optimizer")

        # All groups should have updated value
        values = state_mgr.get_state("threshold")
        for value in values:
            assert value == 1.0

    def test_set_state_optimizer_throws_when_param_already_exists(self):
        """set_state() throws when trying to extend with existing optimizer param."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        # 'lr' already exists
        with pytest.raises(RuntimeError):
            state_mgr.set_state("lr", 0.01, "optimizer")


# =============================================================================
# Show State Test Suite - tests that show_state() enumerates available state correctly
# =============================================================================


class TestShowState:
    """Test show_state() enumeration of available state."""

    def test_show_state_returns_list_of_tuples(self):
        """show_state() returns list of (name, flag) tuples."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        result = state_mgr.show_state()

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_show_state_includes_vital_wrapper_states(self):
        """show_state() includes wrapper_states marked vital."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("vital_metric", 1.0, "vital")

        result = state_mgr.show_state()
        result_dict = {key : value for key, value in result}

        assert "vital_metric" in result_dict
        assert result_dict["vital_metric"] == "vital"

    def test_show_state_includes_optional_wrapper_states(self):
        """show_state() includes wrapper_states marked optional."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("optional_metric", 2.0, "optional")

        result = state_mgr.show_state()
        result_dict = {key : value for key, value in result}

        assert "optional_metric" in result_dict
        assert result_dict["optional_metric"] == "optional"

    def test_show_state_includes_native_optimizer_params(self):
        """show_state() includes native optimizer parameters."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        result = state_mgr.show_state()
        result_dict = {key : value for key, value in result}

        assert "lr" in result_dict
        assert result_dict["lr"] == "optimizer"

    def test_show_state_shows_scalar_tensor_params(self):
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        state_mgr.set_state("custom_param", torch.tensor(3.0), "optimizer")

        result = state_mgr.show_state()
        result_dict = {key: value for key, value in result}

        assert "custom_param" in result_dict
        assert result_dict["custom_param"] == "optimizer"

    def test_show_state_includes_extended_optimizer_params(self):
        """show_state() includes params extended via set_state(..., 'optimizer')."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state_mgr.set_state("custom_param", 1.5, "optimizer")

        result = state_mgr.show_state()
        result_dict = {key : value for key, value in result}

        assert "custom_param" in result_dict
        assert result_dict["custom_param"] == "optimizer"

    def test_show_state_filters_non_numeric_optimizer_params(self):
        """show_state() filters out non-numeric optimizer params."""
        optimizer = create_simple_optimizer()
        # Add non-numeric param
        optimizer.param_groups[0]['string_param'] = "not_numeric"

        state_mgr = StateManagementSubsystem(optimizer)

        result = state_mgr.show_state()
        result_dict = {key : value for key, value in result}

        # Should not include non-numeric param
        assert "string_param" not in result_dict

    def test_show_state_filters_non_scalar_tensor_optimizer_params(self):
        """show_state() filters out non-numeric optimizer params."""
        optimizer = create_simple_optimizer()
        # Add non-numeric param
        optimizer.param_groups[0]['1d_tensor_param'] = torch.tensor([3.0, 5.0, 7.0])

        state_mgr = StateManagementSubsystem(optimizer)

        result = state_mgr.show_state()
        result_dict = {key : value for key, value in result}

        # Should not include non-numeric param
        assert "1d_tensor_param" not in result_dict

    def test_show_state_excludes_params_not_in_all_groups(self):
        """show_state() excludes optimizer params not shared across all groups."""
        optimizer = create_multigroup_optimizer()
        # Add param to only first group
        optimizer.param_groups[0]['group_specific_param'] = 0.5

        state_mgr = StateManagementSubsystem(optimizer)

        result = state_mgr.show_state()
        result_dict = dict(result)

        # Should not include param that's not in all groups
        assert "group_specific_param" not in result_dict


# =============================================================================
# Serialization Test Suite - tests that state_dict/load_state_dict preserve state correctly
# =============================================================================


class TestSerialization:
    """Test state_dict() and load_state_dict() for checkpointing."""

    def test_state_dict_returns_dict(self):
        """state_dict() returns a dictionary."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        state = state_mgr.state_dict()
        assert isinstance(state, dict)


    def test_roundtrip_preserves_vital_wrapper_states(self):
        """state_dict/load_state_dict preserves vital wrapper states."""
        optimizer1 = create_simple_optimizer()
        state_mgr1 = StateManagementSubsystem(optimizer1)

        state_mgr1.set_state("metric1", 42.0, "vital")
        state_mgr1.set_state("metric2", 99.9, "vital")

        state = state_mgr1.state_dict()

        optimizer2 = create_simple_optimizer()
        state_mgr2 = StateManagementSubsystem(optimizer2)
        state_mgr2.load_state_dict(state)

        assert state_mgr2.get_state("metric1") == 42.0
        assert state_mgr2.get_state("metric2") == 99.9

    def test_roundtrip_preserves_optional_wrapper_states(self):
        """state_dict/load_state_dict preserves optional wrapper states."""
        optimizer1 = create_simple_optimizer()
        state_mgr1 = StateManagementSubsystem(optimizer1)

        state_mgr1.set_state("debug_info", "test_value", "optional")

        state = state_mgr1.state_dict()

        optimizer2 = create_simple_optimizer()
        state_mgr2 = StateManagementSubsystem(optimizer2)
        state_mgr2.load_state_dict(state)

        assert state_mgr2.get_state("debug_info") == "test_value"

    def test_roundtrip_preserves_optimizer_state(self):
        """state_dict/load_state_dict preserves optimizer state."""
        optimizer1 = create_simple_optimizer(lr=0.789)
        state_mgr1 = StateManagementSubsystem(optimizer1)

        # Modify optimizer state
        optimizer1.param_groups[0]['lr'] = 0.123

        state = state_mgr1.state_dict()

        optimizer2 = create_simple_optimizer(lr=0.999)  # Different initial lr
        state_mgr2 = StateManagementSubsystem(optimizer2)
        state_mgr2.load_state_dict(state)

        # Default returns list
        lr_list = state_mgr2.get_state("lr")
        assert lr_list == [0.123]

    def test_roundtrip_preserves_extended_optimizer_params(self):
        """state_dict/load_state_dict preserves extended optimizer params."""
        optimizer1 = create_simple_optimizer()
        state_mgr1 = StateManagementSubsystem(optimizer1)

        state_mgr1.set_state("custom_threshold", 0.75, "optimizer")

        state = state_mgr1.state_dict()

        optimizer2 = create_simple_optimizer()
        state_mgr2 = StateManagementSubsystem(optimizer2)
        state_mgr2.load_state_dict(state)

        # Default returns list
        threshold_list = state_mgr2.get_state("custom_threshold")
        assert threshold_list == [0.75]
        # Should also be in param_groups
        assert optimizer2.param_groups[0]["custom_threshold"] == 0.75

    def test_roundtrip_with_mixed_state(self):
        """state_dict/load_state_dict works with all state types together."""
        optimizer1 = create_multigroup_optimizer()
        state_mgr1 = StateManagementSubsystem(optimizer1)

        state_mgr1.set_state("vital1", 1.0, "vital")
        state_mgr1.set_state("vital2", 2.0, "vital")
        state_mgr1.set_state("optional1", "test", "optional")
        state_mgr1.set_state("threshold", 0.5, "optimizer")

        state = state_mgr1.state_dict()

        optimizer2 = create_multigroup_optimizer()
        state_mgr2 = StateManagementSubsystem(optimizer2)
        state_mgr2.load_state_dict(state)

        # Wrapper states return values directly
        assert state_mgr2.get_state("vital1") == 1.0
        assert state_mgr2.get_state("vital2") == 2.0
        assert state_mgr2.get_state("optional1") == "test"
        # Optimizer state returns list (2 param groups)
        threshold_list = state_mgr2.get_state("threshold")
        assert isinstance(threshold_list, list)
        assert len(threshold_list) == 2
        assert all(val == 0.5 for val in threshold_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

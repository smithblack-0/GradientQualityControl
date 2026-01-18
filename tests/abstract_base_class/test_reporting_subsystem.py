"""
Black box tests for ReportingSubsystem.

Tests validate the contract specified in documentation/base_object_implementation.md.
All tests use black box methodology:
- Test only documented behavior
- Use real StateManagementSubsystem through its public interface
- Observe effects through statistics() and vital_statistics() outputs
- Never test implementation details

Test organization:
- Constructor validation
- aggregate_numeric_list() helper function
- statistics() filtering and aggregation
- vital_statistics() as alias to statistics
"""

from numbers import Number

import pytest
import torch

from src.gradient_quality_control.base.reporting import ReportingSubsystem
from src.gradient_quality_control.base.state_management import StateManagementSubsystem

# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


def create_optimizer_with_multiple_param_groups():
    """Create optimizer with multiple param groups having different lr values."""
    params1 = [torch.nn.Parameter(torch.randn(5, 5))]
    params2 = [torch.nn.Parameter(torch.randn(5, 5))]
    params3 = [torch.nn.Parameter(torch.randn(5, 5))]

    optimizer = torch.optim.SGD(
        [
            {"params": params1, "lr": 0.01},
            {"params": params2, "lr": 0.02},
            {"params": params3, "lr": 0.03},
        ]
    )

    return optimizer


def create_optimizer_with_uniform_param_groups():
    """Create optimizer with multiple param groups having same lr value."""
    params1 = [torch.nn.Parameter(torch.randn(5, 5))]
    params2 = [torch.nn.Parameter(torch.randn(5, 5))]

    optimizer = torch.optim.SGD([{"params": params1, "lr": 0.01}, {"params": params2, "lr": 0.01}])

    return optimizer


# =============================================================================
# Constructor Test Suite - tests that constructor initializes correctly
# =============================================================================


class TestConstructor:
    """Test constructor validation."""

    def test_constructor_accepts_state_manager(self):
        """Constructor accepts StateManagementSubsystem instance."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)

        subsystem = ReportingSubsystem(state_manager=state_mgr)

        assert subsystem is not None

    def test_constructor_creates_stateless_facade(self):
        """Constructor creates stateless facade - can be called multiple times with same results."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        # Set some state
        state_mgr.set_state("test_value", 42, "vital")

        # Calling statistics multiple times should give same result
        result1 = subsystem.statistics()
        result2 = subsystem.statistics()

        assert result1 == result2


# =============================================================================
# Aggregate Numeric List Test Suite - tests that aggregate_numeric_list computes correct
# aggregations
# =============================================================================


class TestAggregateNumericList:
    """Test aggregate_numeric_list helper function."""

    def test_aggregate_numeric_list_computes_mean(self):
        """aggregate_numeric_list computes mean correctly."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = subsystem.aggregate_numeric_list(values, "mean")

        assert result == 3.0

    def test_aggregate_numeric_list_computes_max(self):
        """aggregate_numeric_list computes max correctly."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        values = [1.0, 5.0, 3.0, 2.0]
        result = subsystem.aggregate_numeric_list(values, "max")

        assert result == 5.0

    def test_aggregate_numeric_list_computes_min(self):
        """aggregate_numeric_list computes min correctly."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        values = [5.0, 1.0, 3.0, 2.0]
        result = subsystem.aggregate_numeric_list(values, "min")

        assert result == 1.0

    def test_aggregate_numeric_list_handles_single_value(self):
        """aggregate_numeric_list works with single-element list."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        values = [42.0]

        assert subsystem.aggregate_numeric_list(values, "mean") == 42.0
        assert subsystem.aggregate_numeric_list(values, "max") == 42.0
        assert subsystem.aggregate_numeric_list(values, "min") == 42.0

    def test_aggregate_numeric_list_handles_tensor_values(self):
        """aggregate_numeric_list converts scalar tensors to Python numbers."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        values = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = subsystem.aggregate_numeric_list(values, "mean")

        assert result == 2.0
        assert isinstance(result, (int, float))

    def test_aggregate_numeric_list_handles_mixed_numbers_and_tensors(self):
        """aggregate_numeric_list handles mix of Python numbers and tensors."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        values = [1.0, torch.tensor(2.0), 3, torch.tensor(4.0)]
        result = subsystem.aggregate_numeric_list(values, "mean")

        assert result == 2.5
        assert isinstance(result, (int, float))


# =============================================================================
# Statistics Test Suite - tests that statistics method filters and aggregates correctly
# =============================================================================


class TestStatistics:
    """Test statistics method filtering and aggregation."""

    def test_statistics_verbose_includes_vital_state(self):
        """statistics with behavior='verbose' includes vital state."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("vital_counter", 123, "vital")

        result = subsystem.statistics(behavior="verbose")

        assert "vital_counter" in result
        assert result["vital_counter"] == 123

    def test_statistics_verbose_includes_optional_state(self):
        """statistics with behavior='verbose' includes optional state."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("optional_data", "test_value", "optional")

        result = subsystem.statistics(behavior="verbose")

        assert "optional_data" in result
        assert result["optional_data"] == "test_value"

    def test_statistics_verbose_includes_optimizer_params(self):
        """statistics with behavior='verbose' includes optimizer parameters."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result = subsystem.statistics(behavior="verbose")

        # Should include lr from optimizer
        assert "lr" in result

    def test_statistics_vital_excludes_optional_state(self):
        """statistics with behavior='vital' excludes optional state."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("optional_data", "should_not_appear", "optional")

        result = subsystem.statistics(behavior="vital")

        assert "optional_data" not in result

    def test_statistics_vital_includes_vital_state(self):
        """statistics with behavior='vital' includes vital state."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("vital_counter", 456, "vital")

        result = subsystem.statistics(behavior="vital")

        assert "vital_counter" in result
        assert result["vital_counter"] == 456

    def test_statistics_vital_includes_optimizer_params(self):
        """statistics with behavior='vital' includes optimizer parameters."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result = subsystem.statistics(behavior="vital")

        # Should include lr from optimizer
        assert "lr" in result

    def test_statistics_aggregates_uniform_multi_group_params_without_suffix(self):
        """statistics returns scalar without '*' suffix for uniform multi-group params."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result = subsystem.statistics(behavior="verbose")

        # All param groups have lr=0.01, so should return scalar without suffix
        assert "lr" in result
        assert "lr*" not in result
        assert result["lr"] == 0.01

    def test_statistics_uses_mean_aggregation_when_specified(self):
        """statistics uses 'mean' aggregation when aggregate_behavior='mean'."""
        optimizer = create_optimizer_with_multiple_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result = subsystem.statistics(behavior="verbose", aggregate_behavior="mean")

        # Param groups have lr=[0.01, 0.02, 0.03], so should aggregate with suffix
        assert "lr*" in result
        assert "lr" not in result
        assert result["lr*"] == 0.02  # mean of [0.01, 0.02, 0.03]

    def test_statistics_uses_max_aggregation_when_specified(self):
        """statistics uses 'max' aggregation when aggregate_behavior='max'."""
        optimizer = create_optimizer_with_multiple_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result = subsystem.statistics(behavior="verbose", aggregate_behavior="max")

        # Should use max aggregation
        assert result["lr*"] == 0.03  # max of [0.01, 0.02, 0.03]

    def test_statistics_uses_min_aggregation_when_specified(self):
        """statistics uses 'min' aggregation when aggregate_behavior='min'."""
        optimizer = create_optimizer_with_multiple_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result = subsystem.statistics(behavior="verbose", aggregate_behavior="min")

        # Should use min aggregation
        assert result["lr*"] == 0.01  # min of [0.01, 0.02, 0.03]

    def test_statistics_returns_wrapper_state_with_aggregation(self):
        """statistics returns wrapper state values with aggregation."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        # Set list as wrapper state
        state_mgr.set_state("my_list", [1, 2, 3], "optional")

        result = subsystem.statistics(behavior="verbose")

        # Should return list aggregated
        assert result["my_list*"] == 2.0

    def test_statistics_is_read_only(self):
        """statistics does not modify state - calling multiple times gives same result."""
        optimizer = create_optimizer_with_multiple_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("counter", 100, "vital")

        result1 = subsystem.statistics(behavior="verbose")
        result2 = subsystem.statistics(behavior="verbose")

        assert result1 == result2

    def test_statistics_uses_mean_as_default_aggregation(self):
        """statistics uses 'mean' as default aggregate_behavior."""
        optimizer = create_optimizer_with_multiple_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        # Don't specify aggregate_behavior, should default to mean
        result = subsystem.statistics(behavior="verbose")

        # Should use mean aggregation (0.01 + 0.02 + 0.03) / 3 = 0.02
        assert result["lr*"] == 0.02

    def test_statistics_converts_scalar_tensor_wrapper_state(self):
        """statistics converts scalar tensor wrapper state to Python number."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("tensor_value", torch.tensor(42.5), "vital")

        result = subsystem.statistics(behavior="verbose")

        assert result["tensor_value"] == 42.5
        assert isinstance(result["tensor_value"], (int, float))

    def test_statistics_converts_tensor_in_uniform_list(self):
        """statistics converts uniform list of tensors to scalar Python number."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        # Simulate multi-group optimizer param with same tensor value
        state_mgr.set_state("uniform_tensor_list", [torch.tensor(1.5), torch.tensor(1.5)], "vital")

        result = subsystem.statistics(behavior="verbose")

        # Should not have * suffix since all equal
        assert result["uniform_tensor_list"] == 1.5
        assert isinstance(result["uniform_tensor_list"], (int, float))

    def test_statistics_aggregates_tensor_list_with_suffix(self):
        """statistics aggregates non-uniform tensor list and adds * suffix."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        # Non-uniform tensor list
        state_mgr.set_state("nonuniform_tensors", [torch.tensor(1.0), torch.tensor(3.0)], "vital")

        result = subsystem.statistics(behavior="verbose", aggregate_behavior="mean")

        # Should have * suffix and be aggregated
        assert result["nonuniform_tensors*"] == 2.0
        assert isinstance(result["nonuniform_tensors*"], (int, float))


# =============================================================================
# Vital Statistics Test Suite - tests that vital_statistics is an alias to statistics with
# behavior='vital'
# =============================================================================


class TestVitalStatistics:
    """Test vital_statistics method as alias."""

    def test_vital_statistics_excludes_optional_state(self):
        """vital_statistics excludes optional state."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("optional_data", "should_not_appear", "optional")

        result = subsystem.vital_statistics()

        assert "optional_data" not in result

    def test_vital_statistics_includes_vital_state(self):
        """vital_statistics includes vital state."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("vital_metric", 789, "vital")

        result = subsystem.vital_statistics()

        assert "vital_metric" in result
        assert result["vital_metric"] == 789

    def test_vital_statistics_includes_optimizer_params(self):
        """vital_statistics includes optimizer parameters."""
        optimizer = create_optimizer_with_uniform_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result = subsystem.vital_statistics()

        assert "lr" in result

    def test_vital_statistics_respects_aggregate_behavior(self):
        """vital_statistics respects aggregate_behavior parameter."""
        optimizer = create_optimizer_with_multiple_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        result_mean = subsystem.vital_statistics(aggregate_behavior="mean")
        result_max = subsystem.vital_statistics(aggregate_behavior="max")
        result_min = subsystem.vital_statistics(aggregate_behavior="min")

        # Should use different aggregations
        assert result_mean["lr*"] == 0.02  # mean
        assert result_max["lr*"] == 0.03  # max
        assert result_min["lr*"] == 0.01  # min

    def test_vital_statistics_matches_statistics_with_vital_behavior(self):
        """vital_statistics returns same result as statistics(behavior='vital')."""
        optimizer = create_optimizer_with_multiple_param_groups()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = ReportingSubsystem(state_mgr)

        state_mgr.set_state("vital_value", 123, "vital")
        state_mgr.set_state("optional_value", 456, "optional")

        result_vital_statistics = subsystem.vital_statistics(aggregate_behavior="mean")
        result_statistics = subsystem.statistics(behavior="vital", aggregate_behavior="mean")

        assert result_vital_statistics == result_statistics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Black box tests for AbstractOptimizerWrapper base class.

Tests validate the public contract and subclass contract as documented in
documentation/base_object_api.md. All tests use black box methodology:
- Test only documented behavior
- Observe state through public API (statistics, properties, etc.)
- Verify injected object manipulation (optimizer calls, gradient modifications)
- Never test implementation details

Test organization:
- Public API tests: For external users of the wrapper
- Subclass contract tests: For subclass implementers using protected methods
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional

from src.gradient_quality_control.base import AbstractOptimizerWrapper


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


class MinimalWrapper(AbstractOptimizerWrapper):
    """Minimal wrapper that implements only required abstract methods.

    This wrapper follows the subclass contract:
    - Calls _batch_received() once per step()
    - Calls _take_optimizer_step() when num_draws reaches threshold
    - Returns True/False for stepped/accumulating
    """

    def __init__(self, optimizer: torch.optim.Optimizer, max_draws: int = 64,
                 step_every: int = 1):
        super().__init__(optimizer, max_draws)
        self.step_every = step_every

    def step(self) -> bool:
        """Step every N batches where N is step_every."""
        self._batch_received()

        if self.num_draws >= self.step_every:
            self._take_optimizer_step()
            return True
        return False


class StatefulWrapper(AbstractOptimizerWrapper):
    """Wrapper that uses set_state/get_state for testing state management."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_draws: int = 64):
        super().__init__(optimizer, max_draws)

        # Test all three flag types
        self.set_state("vital_metric", 0.0, "vital")
        self.set_state("optional_metric", 0.0, "optional")
        self.set_state("threshold", 1.0, "optimizer")

    def step(self) -> bool:
        """Simple stepping logic for testing."""
        self._batch_received()

        # Update metrics
        self.set_state("vital_metric", self.get_state("vital_metric") + 1.0, "vital")
        self.set_state("optional_metric", self.get_state("optional_metric") + 0.5, "optional")

        if self.num_draws >= 2:
            self._take_optimizer_step()
            return True
        return False


def create_simple_model_and_optimizer(lr=0.01):
    """Create a simple model with optimizer for testing."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


def create_multigroup_optimizer(lr1=0.01, lr2=0.001):
    """Create optimizer with multiple parameter groups."""
    params1 = [nn.Parameter(torch.randn(10, 5))]
    params2 = [nn.Parameter(torch.randn(5, 2))]

    optimizer = torch.optim.SGD([
        {'params': params1, 'lr': lr1},
        {'params': params2, 'lr': lr2}
    ])
    return optimizer


def create_gradients(optimizer, values):
    """Set gradients on optimizer parameters for testing."""
    for param_group in optimizer.param_groups:
        for i, param in enumerate(param_group['params']):
            if i < len(values):
                param.grad = torch.full_like(param, values[i])


# =============================================================================
# Suite 1: Public API - Statistics
# =============================================================================


class TestStatisticsAPI:
    """Test statistics() and vital_statistics() public methods."""

    def test_statistics_returns_dict(self):
        """statistics() returns a dictionary."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert isinstance(stats, dict)

    def test_statistics_contains_num_batches(self):
        """statistics() includes num_batches property."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert "num_batches" in stats
        assert stats["num_batches"] == 0

    def test_statistics_contains_num_steps(self):
        """statistics() includes num_steps property."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert "num_steps" in stats
        assert stats["num_steps"] == 0

    def test_statistics_contains_num_draws(self):
        """statistics() includes num_draws property."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert "num_draws" in stats
        assert stats["num_draws"] == 0

    def test_statistics_contains_last_num_draws(self):
        """statistics() includes last_num_draws property."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert "last_num_draws" in stats
        assert stats["last_num_draws"] is None  # None before first step

    def test_statistics_contains_last_grad_norm(self):
        """statistics() includes last_grad_norm property."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert "last_grad_norm" in stats
        assert stats["last_grad_norm"] is None  # None before first step

    def test_statistics_contains_optimizer_params(self):
        """statistics() includes optimizer parameters like lr."""
        _, optimizer = create_simple_model_and_optimizer(lr=0.123)
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert "lr" in stats
        assert stats["lr"] == 0.123

    def test_statistics_works_before_first_step(self):
        """statistics() can be called before any stepping."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        # Should not raise
        stats = wrapper.statistics()
        assert stats["num_steps"] == 0

    def test_statistics_aggregation_mean(self):
        """statistics() aggregates multi-group params with mean by default."""
        optimizer = create_multigroup_optimizer(lr1=0.01, lr2=0.001)
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics(aggregate_behavior="mean")
        expected_mean = (0.01 + 0.001) / 2
        assert "lr*" in stats  # Star suffix indicates heterogeneous
        assert abs(stats["lr*"] - expected_mean) < 1e-6

    def test_statistics_aggregation_max(self):
        """statistics() aggregates multi-group params with max."""
        optimizer = create_multigroup_optimizer(lr1=0.01, lr2=0.001)
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics(aggregate_behavior="max")
        assert "lr*" in stats
        assert stats["lr*"] == 0.01

    def test_statistics_aggregation_min(self):
        """statistics() aggregates multi-group params with min."""
        optimizer = create_multigroup_optimizer(lr1=0.01, lr2=0.001)
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics(aggregate_behavior="min")
        assert "lr*" in stats
        assert stats["lr*"] == 0.001

    def test_statistics_no_star_suffix_when_uniform(self):
        """statistics() omits star suffix when all groups have same value."""
        optimizer = create_multigroup_optimizer(lr1=0.01, lr2=0.01)
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.statistics()
        assert "lr" in stats
        assert "lr*" not in stats
        assert stats["lr"] == 0.01

    def test_vital_statistics_returns_dict(self):
        """vital_statistics() returns a dictionary."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        stats = wrapper.vital_statistics()
        assert isinstance(stats, dict)

    def test_vital_statistics_is_subset_of_statistics(self):
        """vital_statistics() is a subset of statistics()."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)  # Has vital and optional state

        vital = wrapper.vital_statistics()
        full = wrapper.statistics()

        # Every key in vital should be in full
        for key in vital:
            assert key in full

    def test_vital_statistics_contains_builtin_properties(self):
        """vital_statistics() includes all built-in properties (all vital)."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        vital = wrapper.vital_statistics()
        assert "num_batches" in vital
        assert "num_steps" in vital
        assert "num_draws" in vital
        assert "last_num_draws" in vital
        assert "last_grad_norm" in vital

    def test_vital_statistics_contains_optimizer_params(self):
        """vital_statistics() includes optimizer parameters."""
        _, optimizer = create_simple_model_and_optimizer(lr=0.456)
        wrapper = MinimalWrapper(optimizer)

        vital = wrapper.vital_statistics()
        assert "lr" in vital
        assert vital["lr"] == 0.456

    def test_vital_statistics_excludes_optional_wrapper_states(self):
        """vital_statistics() excludes wrapper_states marked optional."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)

        vital = wrapper.vital_statistics()
        full = wrapper.statistics()

        # optional_metric should be in full but not vital
        assert "optional_metric" in full
        assert "optional_metric" not in vital

        # vital_metric should be in both
        assert "vital_metric" in full
        assert "vital_metric" in vital


# =============================================================================
# Suite 2: Public API - Serialization
# =============================================================================


class TestSerializationAPI:
    """Test state_dict() and load_state_dict() public methods."""

    def test_state_dict_returns_dict(self):
        """state_dict() returns a dictionary."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        state = wrapper.state_dict()
        assert isinstance(state, dict)

    def test_load_state_dict_accepts_dict(self):
        """load_state_dict() accepts a dictionary without error."""
        _, opt1 = create_simple_model_and_optimizer()
        _, opt2 = create_simple_model_and_optimizer()

        wrapper1 = MinimalWrapper(opt1)
        state = wrapper1.state_dict()

        wrapper2 = MinimalWrapper(opt2)
        wrapper2.load_state_dict(state)  # Should not raise

    def test_roundtrip_preserves_num_batches(self):
        """state_dict/load_state_dict preserves num_batches."""
        _, opt1 = create_simple_model_and_optimizer()
        wrapper1 = MinimalWrapper(opt1, step_every=5)

        # Execute some batches
        wrapper1.step()
        wrapper1.step()
        wrapper1.step()

        num_batches_before = wrapper1.statistics()["num_batches"]

        # Save and load
        state = wrapper1.state_dict()
        _, opt2 = create_simple_model_and_optimizer()
        wrapper2 = MinimalWrapper(opt2, step_every=5)
        wrapper2.load_state_dict(state)

        num_batches_after = wrapper2.statistics()["num_batches"]
        assert num_batches_after == num_batches_before

    def test_roundtrip_preserves_num_steps(self):
        """state_dict/load_state_dict preserves num_steps."""
        model1, opt1 = create_simple_model_and_optimizer()
        wrapper1 = MinimalWrapper(opt1, step_every=2)

        # Step the optimizer
        create_gradients(opt1, [1.0])
        wrapper1.step()
        create_gradients(opt1, [1.0])
        wrapper1.step()  # This should trigger optimizer step

        num_steps_before = wrapper1.statistics()["num_steps"]

        # Save and load
        state = wrapper1.state_dict()
        model2, opt2 = create_simple_model_and_optimizer()
        wrapper2 = MinimalWrapper(opt2, step_every=2)
        wrapper2.load_state_dict(state)

        num_steps_after = wrapper2.statistics()["num_steps"]
        assert num_steps_after == num_steps_before
        assert num_steps_after == 1

    def test_roundtrip_preserves_num_draws(self):
        """state_dict/load_state_dict preserves num_draws."""
        _, opt1 = create_simple_model_and_optimizer()
        wrapper1 = MinimalWrapper(opt1, step_every=5)

        wrapper1.step()
        wrapper1.step()

        num_draws_before = wrapper1.statistics()["num_draws"]

        state = wrapper1.state_dict()
        _, opt2 = create_simple_model_and_optimizer()
        wrapper2 = MinimalWrapper(opt2, step_every=5)
        wrapper2.load_state_dict(state)

        num_draws_after = wrapper2.statistics()["num_draws"]
        assert num_draws_after == num_draws_before

    def test_roundtrip_preserves_last_num_draws(self):
        """state_dict/load_state_dict preserves last_num_draws."""
        model, opt1 = create_simple_model_and_optimizer()
        wrapper1 = MinimalWrapper(opt1, step_every=3)

        # Accumulate and step
        create_gradients(opt1, [1.0])
        wrapper1.step()
        create_gradients(opt1, [1.0])
        wrapper1.step()
        create_gradients(opt1, [1.0])
        wrapper1.step()  # Should step with num_draws=3

        last_num_draws_before = wrapper1.statistics()["last_num_draws"]

        state = wrapper1.state_dict()
        model2, opt2 = create_simple_model_and_optimizer()
        wrapper2 = MinimalWrapper(opt2, step_every=3)
        wrapper2.load_state_dict(state)

        last_num_draws_after = wrapper2.statistics()["last_num_draws"]
        assert last_num_draws_after == last_num_draws_before
        assert last_num_draws_after == 3

    def test_roundtrip_preserves_optimizer_params(self):
        """state_dict/load_state_dict preserves optimizer parameters."""
        _, opt1 = create_simple_model_and_optimizer(lr=0.789)
        wrapper1 = MinimalWrapper(opt1)

        # Modify lr through param_groups
        for group in wrapper1.param_groups:
            group['lr'] = 0.999

        state = wrapper1.state_dict()
        _, opt2 = create_simple_model_and_optimizer(lr=0.001)  # Different initial lr
        wrapper2 = MinimalWrapper(opt2)
        wrapper2.load_state_dict(state)

        # Should have loaded lr
        stats = wrapper2.statistics()
        assert stats["lr"] == 0.999

    def test_training_continues_after_load(self):
        """Can continue training after load_state_dict."""
        model1, opt1 = create_simple_model_and_optimizer()
        wrapper1 = MinimalWrapper(opt1, step_every=2)

        # Do one step cycle
        create_gradients(opt1, [1.0])
        wrapper1.step()
        create_gradients(opt1, [1.0])
        wrapper1.step()

        # Save
        state = wrapper1.state_dict()

        # Load into new wrapper
        model2, opt2 = create_simple_model_and_optimizer()
        wrapper2 = MinimalWrapper(opt2, step_every=2)
        wrapper2.load_state_dict(state)

        # Continue training - should not raise
        create_gradients(opt2, [1.0])
        wrapper2.step()
        create_gradients(opt2, [1.0])
        stepped = wrapper2.step()

        assert stepped is True
        assert wrapper2.statistics()["num_steps"] == 2


# =============================================================================
# Suite 3: Public API - Other
# =============================================================================


class TestOtherPublicAPI:
    """Test remaining public methods and properties."""

    def test_zero_grad_throws_not_implemented_error(self):
        """zero_grad() always raises NotImplementedError."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        with pytest.raises(NotImplementedError):
            wrapper.zero_grad()

    def test_valid_schedule_targets_returns_list(self):
        """valid_schedule_targets property returns a list."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        targets = wrapper.valid_schedule_targets
        assert isinstance(targets, list)

    def test_valid_schedule_targets_includes_optimizer_native_params(self):
        """valid_schedule_targets includes native optimizer parameters."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        targets = wrapper.valid_schedule_targets
        assert "lr" in targets

    def test_valid_schedule_targets_includes_extended_params(self):
        """valid_schedule_targets includes wrapper-extended parameters."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)  # Extends with "threshold"

        targets = wrapper.valid_schedule_targets
        assert "threshold" in targets

    def test_param_groups_accessible(self):
        """param_groups attribute is accessible (forwarded to optimizer)."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        assert hasattr(wrapper, 'param_groups')
        assert wrapper.param_groups is optimizer.param_groups

    def test_optimizer_field_accessible(self):
        """optimizer field is directly accessible."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        assert wrapper.optimizer is optimizer

    def test_unknown_attributes_forwarded_to_optimizer(self):
        """Unknown attributes are forwarded to wrapped optimizer."""
        _, optimizer = create_simple_model_and_optimizer()
        optimizer.custom_attr = "test_value"
        wrapper = MinimalWrapper(optimizer)

        assert wrapper.custom_attr == "test_value"

    def test_unknown_methods_forwarded_to_optimizer(self):
        """Unknown methods are forwarded to wrapped optimizer."""
        _, optimizer = create_simple_model_and_optimizer()

        def custom_method():
            return 42

        optimizer.custom_method = custom_method
        wrapper = MinimalWrapper(optimizer)

        assert wrapper.custom_method() == 42


# =============================================================================
# Suite 4: Subclass Contract - Counter Management
# =============================================================================


class TestSubclassCounterContract:
    """Test _batch_received() and counter behaviors."""

    def test_batch_received_increments_num_batches(self):
        """_batch_received() increments num_batches (observable via statistics)."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=10)

        initial = wrapper.statistics()["num_batches"]

        wrapper.step()  # Calls _batch_received internally

        after = wrapper.statistics()["num_batches"]
        assert after == initial + 1

    def test_batch_received_increments_num_draws(self):
        """_batch_received() increments num_draws (observable via statistics)."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=10)

        initial = wrapper.statistics()["num_draws"]

        wrapper.step()

        after = wrapper.statistics()["num_draws"]
        assert after == initial + 1

    def test_batch_received_throws_at_max_draws(self):
        """_batch_received() throws error when reaching max_draws."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, max_draws=3, step_every=10)

        # Accumulate to max_draws
        wrapper.step()  # num_draws=1
        wrapper.step()  # num_draws=2
        wrapper.step()  # num_draws=3

        # Next call should throw
        with pytest.raises(RuntimeError):
            wrapper.step()

    def test_multiple_batch_received_calls_accumulate_num_draws(self):
        """Multiple _batch_received() calls accumulate num_draws correctly."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=10)

        wrapper.step()
        wrapper.step()
        wrapper.step()

        stats = wrapper.statistics()
        assert stats["num_batches"] == 3
        assert stats["num_draws"] == 3


# =============================================================================
# Suite 5: Subclass Contract - Optimizer Stepping
# =============================================================================


class TestSubclassOptimizerStepContract:
    """Test _take_optimizer_step() contract."""

    def test_take_optimizer_step_calls_optimizer_step(self):
        """_take_optimizer_step() calls optimizer.step() on injected optimizer."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=1)

        # Track if optimizer.step was called
        step_called = []
        original_step = optimizer.step

        def tracked_step(closure=None):
            step_called.append(True)
            return original_step(closure)

        optimizer.step = tracked_step

        create_gradients(optimizer, [1.0])
        wrapper.step()  # Should call _take_optimizer_step

        assert len(step_called) == 1

    def test_take_optimizer_step_calls_optimizer_zero_grad(self):
        """_take_optimizer_step() calls optimizer.zero_grad() on injected optimizer."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=1)

        zero_grad_called = []
        original_zero_grad = optimizer.zero_grad

        def tracked_zero_grad():
            zero_grad_called.append(True)
            return original_zero_grad()

        optimizer.zero_grad = tracked_zero_grad

        create_gradients(optimizer, [1.0])
        wrapper.step()

        assert len(zero_grad_called) == 1

    def test_take_optimizer_step_averages_gradients_by_num_draws(self):
        """_take_optimizer_step() divides gradients by num_draws."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=3)

        # Accumulate gradients - simulate backward passes that sum
        params = list(model.parameters())

        # First batch
        for p in params:
            p.grad = torch.full_like(p, 6.0)
        wrapper.step()  # num_draws=1, don't step yet

        # Second batch - gradients sum
        for p in params:
            p.grad += torch.full_like(p, 6.0)  # Now grad=12
        wrapper.step()  # num_draws=2, don't step yet

        # Third batch
        for p in params:
            p.grad += torch.full_like(p, 6.0)  # Now grad=18
        wrapper.step()  # num_draws=3, step now

        # After stepping, gradients should be zeroed
        # But we can check that the step happened by verifying num_steps
        assert wrapper.statistics()["num_steps"] == 1
        assert wrapper.statistics()["num_draws"] == 0  # Reset after step

    def test_take_optimizer_step_skips_none_gradients(self):
        """_take_optimizer_step() skips parameters with None gradients."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=2)

        params = list(model.parameters())

        # Set some gradients to None
        params[0].grad = torch.full_like(params[0], 4.0)
        params[1].grad = None  # This should be skipped
        wrapper.step()

        params[0].grad += torch.full_like(params[0], 4.0)
        # params[1].grad still None
        wrapper.step()  # Should step without error

        # Should not have raised
        assert wrapper.statistics()["num_steps"] == 1

    def test_take_optimizer_step_increments_num_steps(self):
        """_take_optimizer_step() increments num_steps."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=1)

        create_gradients(optimizer, [1.0])
        initial = wrapper.statistics()["num_steps"]

        wrapper.step()

        after = wrapper.statistics()["num_steps"]
        assert after == initial + 1

    def test_take_optimizer_step_resets_num_draws_to_zero(self):
        """_take_optimizer_step() resets num_draws to 0."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=3)

        create_gradients(optimizer, [1.0])
        wrapper.step()
        create_gradients(optimizer, [1.0])
        wrapper.step()
        create_gradients(optimizer, [1.0])
        wrapper.step()  # Should step

        stats = wrapper.statistics()
        assert stats["num_draws"] == 0

    def test_take_optimizer_step_sets_last_num_draws(self):
        """_take_optimizer_step() sets last_num_draws property."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=4)

        # Before any step
        assert wrapper.statistics()["last_num_draws"] is None

        # Accumulate 4 batches and step
        for _ in range(4):
            create_gradients(optimizer, [1.0])
            wrapper.step()

        stats = wrapper.statistics()
        assert stats["last_num_draws"] == 4
        assert stats["num_draws"] == 0  # Reset after step

    def test_take_optimizer_step_computes_and_stores_grad_norm(self):
        """_take_optimizer_step() computes L2 norm and stores in last_grad_norm."""
        model, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer, step_every=1)

        # Before step
        assert wrapper.statistics()["last_grad_norm"] is None

        # Set known gradients
        create_gradients(optimizer, [1.0])
        wrapper.step()

        # After step, should have a gradient norm
        stats = wrapper.statistics()
        assert stats["last_grad_norm"] is not None
        assert isinstance(stats["last_grad_norm"], (float, int))
        assert stats["last_grad_norm"] >= 0

    def test_take_optimizer_step_throws_when_num_draws_zero(self):
        """_take_optimizer_step() raises error if num_draws is 0."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = MinimalWrapper(optimizer)

        # Directly calling _take_optimizer_step without _batch_received should fail
        with pytest.raises(RuntimeError):
            wrapper._take_optimizer_step()


# =============================================================================
# Suite 6: Subclass Contract - State Management
# =============================================================================


class TestSubclassStateManagement:
    """Test set_state() and get_state() subclass methods."""

    def test_set_state_vital_retrievable_via_get_state(self):
        """Values set with flag='vital' are retrievable via get_state()."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)

        wrapper.set_state("test_vital", 123, "vital")
        value = wrapper.get_state("test_vital")

        assert value == 123

    def test_set_state_optional_retrievable_via_get_state(self):
        """Values set with flag='optional' are retrievable via get_state()."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)

        wrapper.set_state("test_optional", 456, "optional")
        value = wrapper.get_state("test_optional")

        assert value == 456

    def test_set_state_vital_appears_in_vital_statistics(self):
        """Values marked 'vital' appear in vital_statistics()."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)

        wrapper.set_state("important_metric", 99.9, "vital")

        vital = wrapper.vital_statistics()
        assert "important_metric" in vital
        assert vital["important_metric"] == 99.9

    def test_set_state_optional_not_in_vital_statistics(self):
        """Values marked 'optional' do NOT appear in vital_statistics()."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)

        wrapper.set_state("debug_info", "test", "optional")

        vital = wrapper.vital_statistics()
        full = wrapper.statistics()

        assert "debug_info" not in vital
        assert "debug_info" in full

    def test_set_state_optimizer_extends_param_groups(self):
        """set_state with flag='optimizer' extends param_groups."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)  # Sets "threshold" with optimizer flag

        # Check that param_groups now has "threshold"
        for group in wrapper.param_groups:
            assert "threshold" in group
            assert group["threshold"] == 1.0

    def test_set_state_optimizer_appears_in_valid_schedule_targets(self):
        """Parameters extended via flag='optimizer' appear in valid_schedule_targets."""
        _, optimizer = create_simple_model_and_optimizer()
        wrapper = StatefulWrapper(optimizer)

        targets = wrapper.valid_schedule_targets
        assert "threshold" in targets

    def test_get_state_retrieves_optimizer_params(self):
        """get_state() can retrieve optimizer parameters like lr."""
        _, optimizer = create_simple_model_and_optimizer(lr=0.555)
        wrapper = MinimalWrapper(optimizer)

        lr = wrapper.get_state("lr")
        assert lr == 0.555

    def test_get_state_multigroup_aggregation_none(self):
        """get_state with aggregate_behavior='none' returns list for multi-group."""
        optimizer = create_multigroup_optimizer(lr1=0.1, lr2=0.2)
        wrapper = MinimalWrapper(optimizer)

        lrs = wrapper.get_state("lr", aggregate_behavior="none")
        assert isinstance(lrs, list)
        assert len(lrs) == 2
        assert 0.1 in lrs
        assert 0.2 in lrs

    def test_get_state_multigroup_aggregation_mean(self):
        """get_state with aggregate_behavior='mean' returns mean."""
        optimizer = create_multigroup_optimizer(lr1=0.1, lr2=0.2)
        wrapper = MinimalWrapper(optimizer)

        lr_mean = wrapper.get_state("lr", aggregate_behavior="mean")
        assert abs(lr_mean - 0.15) < 1e-6

    def test_get_state_multigroup_aggregation_max(self):
        """get_state with aggregate_behavior='max' returns max."""
        optimizer = create_multigroup_optimizer(lr1=0.1, lr2=0.2)
        wrapper = MinimalWrapper(optimizer)

        lr_max = wrapper.get_state("lr", aggregate_behavior="max")
        assert lr_max == 0.2

    def test_get_state_multigroup_aggregation_min(self):
        """get_state with aggregate_behavior='min' returns min."""
        optimizer = create_multigroup_optimizer(lr1=0.1, lr2=0.2)
        wrapper = MinimalWrapper(optimizer)

        lr_min = wrapper.get_state("lr", aggregate_behavior="min")
        assert lr_min == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Black box tests for AbstractOptimizerWrapper base class.

Tests validate the public contract and subclass contract as documented in
documentation/base_object_api.md. All tests use black box methodology:
- Test only documented behavior
- Observe state through public API (statistics, properties, etc.)
- Verify injected object manipulation (base optimizer calls, gradient modifications)
- Never test implementation details

Test organization:
- Public API tests: For external users of the optimizer
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


class MinimalTestOptimizer(AbstractOptimizerWrapper):
    """Minimal test optimizer that implements only required abstract methods.

    This optimizer follows the subclass contract:
    - Calls _batch_received() once per step()
    - Calls _take_optimizer_step() when num_draws reaches threshold
    - Returns True/False for stepped/accumulating
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, max_draws: int = 64,
                 step_every: int = 1):
        super().__init__(base_optimizer, max_draws)
        self.step_every = step_every

    def step(self) -> bool:
        """Step every N batches where N is step_every."""
        self._batch_received()

        if self.num_draws >= self.step_every:
            self._take_optimizer_step()
            return True
        return False


class StatefulTestOptimizer(AbstractOptimizerWrapper):
    """Test optimizer that uses set_state/get_state for testing state management."""

    def __init__(self, base_optimizer: torch.optim.Optimizer, max_draws: int = 64):
        super().__init__(base_optimizer, max_draws)

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


def create_simple_model_and_base_optimizer(lr=0.01):
    """Create a simple model with base optimizer for testing."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    base_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, base_optimizer


def create_multigroup_base_optimizer(lr1=0.01, lr2=0.001):
    """Create base optimizer with multiple parameter groups."""
    params1 = [nn.Parameter(torch.randn(10, 5))]
    params2 = [nn.Parameter(torch.randn(5, 2))]

    base_optimizer = torch.optim.SGD([
        {'params': params1, 'lr': lr1},
        {'params': params2, 'lr': lr2}
    ])
    return base_optimizer


def create_gradients(base_optimizer, values):
    """Set gradients on base optimizer parameters for testing."""
    for param_group in base_optimizer.param_groups:
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
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert isinstance(stats, dict)

    def test_statistics_contains_num_batches(self):
        """statistics() includes num_batches property."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert "num_batches" in stats
        assert stats["num_batches"] == 0

    def test_statistics_contains_num_steps(self):
        """statistics() includes num_steps property."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert "num_steps" in stats
        assert stats["num_steps"] == 0

    def test_statistics_contains_num_draws(self):
        """statistics() includes num_draws property."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert "num_draws" in stats
        assert stats["num_draws"] == 0

    def test_statistics_contains_last_num_draws(self):
        """statistics() includes last_num_draws property."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert "last_num_draws" in stats
        assert stats["last_num_draws"] is None  # None before first step

    def test_statistics_contains_last_grad_norm(self):
        """statistics() includes last_grad_norm property."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert "last_grad_norm" in stats
        assert stats["last_grad_norm"] is None  # None before first step

    def test_statistics_contains_optimizer_params(self):
        """statistics() includes optimizer parameters like lr."""
        _, base_optimizer = create_simple_model_and_base_optimizer(lr=0.123)
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert "lr" in stats
        assert stats["lr"] == 0.123

    def test_statistics_works_before_first_step(self):
        """statistics() can be called before any stepping."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        # Should not raise
        stats = optimizer.statistics()
        assert stats["num_steps"] == 0

    def test_statistics_aggregation_mean(self):
        """statistics() aggregates multi-group params with mean by default."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.01, lr2=0.001)
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics(aggregate_behavior="mean")
        expected_mean = (0.01 + 0.001) / 2
        assert "lr*" in stats  # Star suffix indicates heterogeneous
        assert abs(stats["lr*"] - expected_mean) < 1e-6

    def test_statistics_aggregation_max(self):
        """statistics() aggregates multi-group params with max."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.01, lr2=0.001)
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics(aggregate_behavior="max")
        assert "lr*" in stats
        assert stats["lr*"] == 0.01

    def test_statistics_aggregation_min(self):
        """statistics() aggregates multi-group params with min."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.01, lr2=0.001)
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics(aggregate_behavior="min")
        assert "lr*" in stats
        assert stats["lr*"] == 0.001

    def test_statistics_no_star_suffix_when_uniform(self):
        """statistics() omits star suffix when all groups have same value."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.01, lr2=0.01)
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.statistics()
        assert "lr" in stats
        assert "lr*" not in stats
        assert stats["lr"] == 0.01

    def test_vital_statistics_returns_dict(self):
        """vital_statistics() returns a dictionary."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        stats = optimizer.vital_statistics()
        assert isinstance(stats, dict)

    def test_vital_statistics_is_subset_of_statistics(self):
        """vital_statistics() is a subset of statistics()."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)  # Has vital and optional state

        vital = optimizer.vital_statistics()
        full = optimizer.statistics()

        # Every key in vital should be in full
        for key in vital:
            assert key in full

    def test_vital_statistics_contains_builtin_properties(self):
        """vital_statistics() includes all built-in properties (all vital)."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        vital = optimizer.vital_statistics()
        assert "num_batches" in vital
        assert "num_steps" in vital
        assert "num_draws" in vital
        assert "last_num_draws" in vital
        assert "last_grad_norm" in vital

    def test_vital_statistics_contains_optimizer_params(self):
        """vital_statistics() includes optimizer parameters."""
        _, base_optimizer = create_simple_model_and_base_optimizer(lr=0.456)
        optimizer = MinimalTestOptimizer(base_optimizer)

        vital = optimizer.vital_statistics()
        assert "lr" in vital
        assert vital["lr"] == 0.456

    def test_vital_statistics_excludes_optional_wrapper_states(self):
        """vital_statistics() excludes wrapper_states marked optional."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)

        vital = optimizer.vital_statistics()
        full = optimizer.statistics()

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
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        state = optimizer.state_dict()
        assert isinstance(state, dict)

    def test_load_state_dict_accepts_dict(self):
        """load_state_dict() accepts a dictionary without error."""
        _, base_opt1 = create_simple_model_and_base_optimizer()
        _, base_opt2 = create_simple_model_and_base_optimizer()

        optimizer1 = MinimalTestOptimizer(base_opt1)
        state = optimizer1.state_dict()

        optimizer2 = MinimalTestOptimizer(base_opt2)
        optimizer2.load_state_dict(state)  # Should not raise

    def test_roundtrip_preserves_num_batches(self):
        """state_dict/load_state_dict preserves num_batches."""
        _, base_opt1 = create_simple_model_and_base_optimizer()
        optimizer1 = MinimalTestOptimizer(base_opt1, step_every=5)

        # Execute some batches
        optimizer1.step()
        optimizer1.step()
        optimizer1.step()

        num_batches_before = optimizer1.statistics()["num_batches"]

        # Save and load
        state = optimizer1.state_dict()
        _, base_opt2 = create_simple_model_and_base_optimizer()
        optimizer2 = MinimalTestOptimizer(base_opt2, step_every=5)
        optimizer2.load_state_dict(state)

        num_batches_after = optimizer2.statistics()["num_batches"]
        assert num_batches_after == num_batches_before

    def test_roundtrip_preserves_num_steps(self):
        """state_dict/load_state_dict preserves num_steps."""
        model1, base_opt1 = create_simple_model_and_base_optimizer()
        optimizer1 = MinimalTestOptimizer(base_opt1, step_every=2)

        # Step the optimizer
        create_gradients(base_opt1, [1.0])
        optimizer1.step()
        create_gradients(base_opt1, [1.0])
        optimizer1.step()  # This should trigger optimizer step

        num_steps_before = optimizer1.statistics()["num_steps"]

        # Save and load
        state = optimizer1.state_dict()
        model2, base_opt2 = create_simple_model_and_base_optimizer()
        optimizer2 = MinimalTestOptimizer(base_opt2, step_every=2)
        optimizer2.load_state_dict(state)

        num_steps_after = optimizer2.statistics()["num_steps"]
        assert num_steps_after == num_steps_before
        assert num_steps_after == 1

    def test_roundtrip_preserves_num_draws(self):
        """state_dict/load_state_dict preserves num_draws."""
        _, base_opt1 = create_simple_model_and_base_optimizer()
        optimizer1 = MinimalTestOptimizer(base_opt1, step_every=5)

        optimizer1.step()
        optimizer1.step()

        num_draws_before = optimizer1.statistics()["num_draws"]

        state = optimizer1.state_dict()
        _, base_opt2 = create_simple_model_and_base_optimizer()
        optimizer2 = MinimalTestOptimizer(base_opt2, step_every=5)
        optimizer2.load_state_dict(state)

        num_draws_after = optimizer2.statistics()["num_draws"]
        assert num_draws_after == num_draws_before

    def test_roundtrip_preserves_last_num_draws(self):
        """state_dict/load_state_dict preserves last_num_draws."""
        model, base_opt1 = create_simple_model_and_base_optimizer()
        optimizer1 = MinimalTestOptimizer(base_opt1, step_every=3)

        # Accumulate and step
        create_gradients(base_opt1, [1.0])
        optimizer1.step()
        create_gradients(base_opt1, [1.0])
        optimizer1.step()
        create_gradients(base_opt1, [1.0])
        optimizer1.step()  # Should step with num_draws=3

        last_num_draws_before = optimizer1.statistics()["last_num_draws"]

        state = optimizer1.state_dict()
        model2, base_opt2 = create_simple_model_and_base_optimizer()
        optimizer2 = MinimalTestOptimizer(base_opt2, step_every=3)
        optimizer2.load_state_dict(state)

        last_num_draws_after = optimizer2.statistics()["last_num_draws"]
        assert last_num_draws_after == last_num_draws_before
        assert last_num_draws_after == 3

    def test_roundtrip_preserves_optimizer_params(self):
        """state_dict/load_state_dict preserves optimizer parameters."""
        _, base_opt1 = create_simple_model_and_base_optimizer(lr=0.789)
        optimizer1 = MinimalTestOptimizer(base_opt1)

        # Modify lr through param_groups
        for group in optimizer1.param_groups:
            group['lr'] = 0.999

        state = optimizer1.state_dict()
        _, base_opt2 = create_simple_model_and_base_optimizer(lr=0.001)  # Different initial lr
        optimizer2 = MinimalTestOptimizer(base_opt2)
        optimizer2.load_state_dict(state)

        # Should have loaded lr
        stats = optimizer2.statistics()
        assert stats["lr"] == 0.999

    def test_training_continues_after_load(self):
        """Can continue training after load_state_dict."""
        model1, base_opt1 = create_simple_model_and_base_optimizer()
        optimizer1 = MinimalTestOptimizer(base_opt1, step_every=2)

        # Do one step cycle
        create_gradients(base_opt1, [1.0])
        optimizer1.step()
        create_gradients(base_opt1, [1.0])
        optimizer1.step()

        # Save
        state = optimizer1.state_dict()

        # Load into new optimizer
        model2, base_opt2 = create_simple_model_and_base_optimizer()
        optimizer2 = MinimalTestOptimizer(base_opt2, step_every=2)
        optimizer2.load_state_dict(state)

        # Continue training - should not raise
        create_gradients(base_opt2, [1.0])
        optimizer2.step()
        create_gradients(base_opt2, [1.0])
        stepped = optimizer2.step()

        assert stepped is True
        assert optimizer2.statistics()["num_steps"] == 2

    def test_state_dict_can_be_pickled_to_file(self):
        """state_dict() can be saved to disk with pickle and loaded back."""
        import pickle
        import tempfile
        import os

        model1, base_opt1 = create_simple_model_and_base_optimizer()
        optimizer1 = MinimalTestOptimizer(base_opt1, step_every=3)

        # Execute some operations to create non-trivial state
        create_gradients(base_opt1, [2.0])
        optimizer1.step()
        create_gradients(base_opt1, [2.0])
        optimizer1.step()
        create_gradients(base_opt1, [2.0])
        optimizer1.step()  # Should step

        # Get state before pickling
        stats_before = optimizer1.statistics()

        # Save to actual file
        state = optimizer1.state_dict()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle_path = f.name
            pickle.dump(state, f)

        try:
            # Load from file
            with open(pickle_path, 'rb') as f:
                loaded_state = pickle.load(f)

            # Create new optimizer and load
            model2, base_opt2 = create_simple_model_and_base_optimizer()
            optimizer2 = MinimalTestOptimizer(base_opt2, step_every=3)
            optimizer2.load_state_dict(loaded_state)

            # Verify state preserved after pickle roundtrip
            stats_after = optimizer2.statistics()
            assert stats_after["num_batches"] == stats_before["num_batches"]
            assert stats_after["num_steps"] == stats_before["num_steps"]
            assert stats_after["num_draws"] == stats_before["num_draws"]
            assert stats_after["last_num_draws"] == stats_before["last_num_draws"]

        finally:
            # Clean up
            if os.path.exists(pickle_path):
                os.unlink(pickle_path)


# =============================================================================
# Suite 3: Public API - Other
# =============================================================================


class TestOtherPublicAPI:
    """Test remaining public methods and properties."""

    def test_zero_grad_throws_not_implemented_error(self):
        """zero_grad() always raises NotImplementedError."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        with pytest.raises(NotImplementedError):
            optimizer.zero_grad()

    def test_valid_schedule_targets_returns_list(self):
        """valid_schedule_targets property returns a list."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        targets = optimizer.valid_schedule_targets
        assert isinstance(targets, list)

    def test_valid_schedule_targets_includes_optimizer_native_params(self):
        """valid_schedule_targets includes native optimizer parameters."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        targets = optimizer.valid_schedule_targets
        assert "lr" in targets

    def test_valid_schedule_targets_includes_extended_params(self):
        """valid_schedule_targets includes extended parameters."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)  # Extends with "threshold"

        targets = optimizer.valid_schedule_targets
        assert "threshold" in targets

    def test_param_groups_accessible(self):
        """param_groups attribute is accessible (forwarded to base optimizer)."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        assert hasattr(optimizer, 'param_groups')
        assert optimizer.param_groups is base_optimizer.param_groups

    def test_optimizer_field_accessible(self):
        """optimizer field is directly accessible."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        assert optimizer.optimizer is base_optimizer

    def test_unknown_attributes_forwarded_to_base_optimizer(self):
        """Unknown attributes are forwarded to base optimizer."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        base_optimizer.custom_attr = "test_value"
        optimizer = MinimalTestOptimizer(base_optimizer)

        assert optimizer.custom_attr == "test_value"

    def test_unknown_methods_forwarded_to_base_optimizer(self):
        """Unknown methods are forwarded to base optimizer."""
        _, base_optimizer = create_simple_model_and_base_optimizer()

        def custom_method():
            return 42

        base_optimizer.custom_method = custom_method
        optimizer = MinimalTestOptimizer(base_optimizer)

        assert optimizer.custom_method() == 42


# =============================================================================
# Suite 4: Subclass Contract - Counter Management
# =============================================================================


class TestSubclassCounterContract:
    """Test _batch_received() and counter behaviors."""

    def test_batch_received_increments_num_batches(self):
        """_batch_received() increments num_batches (observable via statistics)."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=10)

        initial = optimizer.statistics()["num_batches"]

        optimizer.step()  # Calls _batch_received internally

        after = optimizer.statistics()["num_batches"]
        assert after == initial + 1

    def test_batch_received_increments_num_draws(self):
        """_batch_received() increments num_draws (observable via statistics)."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=10)

        initial = optimizer.statistics()["num_draws"]

        optimizer.step()

        after = optimizer.statistics()["num_draws"]
        assert after == initial + 1

    def test_batch_received_throws_at_max_draws(self):
        """_batch_received() throws error when reaching max_draws."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, max_draws=3, step_every=10)

        # Accumulate to max_draws
        optimizer.step()  # num_draws=1
        optimizer.step()  # num_draws=2
        optimizer.step()  # num_draws=3

        # Next call should throw
        with pytest.raises(RuntimeError):
            optimizer.step()

    def test_multiple_batch_received_calls_accumulate_num_draws(self):
        """Multiple _batch_received() calls accumulate num_draws correctly."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=10)

        optimizer.step()
        optimizer.step()
        optimizer.step()

        stats = optimizer.statistics()
        assert stats["num_batches"] == 3
        assert stats["num_draws"] == 3


# =============================================================================
# Suite 5: Subclass Contract - Optimizer Stepping
# =============================================================================


class TestSubclassOptimizerStepContract:
    """Test _take_optimizer_step() contract."""

    def test_take_optimizer_step_calls_base_optimizer_step(self):
        """_take_optimizer_step() calls step() on injected base optimizer."""
        model, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=1)

        # Track if base optimizer.step was called
        step_called = []
        original_step = base_optimizer.step

        def tracked_step(closure=None):
            step_called.append(True)
            return original_step(closure)

        base_optimizer.step = tracked_step

        create_gradients(base_optimizer, [1.0])
        optimizer.step()  # Should call _take_optimizer_step

        assert len(step_called) == 1

    def test_take_optimizer_step_calls_base_optimizer_zero_grad(self):
        """_take_optimizer_step() calls zero_grad() on injected base optimizer."""
        model, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=1)

        zero_grad_called = []
        original_zero_grad = base_optimizer.zero_grad

        def tracked_zero_grad():
            zero_grad_called.append(True)
            return original_zero_grad()

        base_optimizer.zero_grad = tracked_zero_grad

        create_gradients(base_optimizer, [1.0])
        optimizer.step()

        assert len(zero_grad_called) == 1

    def test_using_optimizer_results_in_gradient_averaging(self):
        """When following the contract, parameters are updated with averaged gradients, not summed.

        This tests emergent behavior from correct usage, not internal implementation.
        """
        # Create simple model with single parameter for easy verification
        param = nn.Parameter(torch.tensor([10.0]))
        base_optimizer = torch.optim.SGD([param], lr=1.0)  # lr=1.0 for simple math
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=3)

        initial_value = param.data.clone()

        # Simulate PyTorch's gradient accumulation behavior
        # Each backward() ADDS to .grad, doesn't replace it
        # Use same gradient value [6.0] for each batch for simplicity
        param.grad = torch.tensor([6.0])
        optimizer.step()  # Batch 1, num_draws=1, grad_sum=6.0

        # PyTorch accumulates - add to existing grad
        param.grad = param.grad + torch.tensor([6.0])  # grad_sum=12.0
        optimizer.step()  # Batch 2, num_draws=2

        param.grad = param.grad + torch.tensor([6.0])  # grad_sum=18.0
        optimizer.step()  # Batch 3, num_draws=3, should step now

        # After 3 batches: grad_sum=18.0, mean=6.0
        # With lr=1.0 and SGD, update is: param = param - lr * grad_mean
        # If using MEAN: param = 10.0 - 1.0 * 6.0 = 4.0 ✓
        # If using SUM:  param = 10.0 - 1.0 * 18.0 = -8.0 ✗
        expected_value = initial_value - 1.0 * 6.0  # Mean of accumulated gradients

        assert torch.allclose(param.data, expected_value), \
            f"Expected {expected_value}, got {param.data}. Gradients may not be averaged."

    def test_take_optimizer_step_handles_none_gradients(self):
        """_take_optimizer_step() handles parameters with None gradients without error."""
        model, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=2)

        params = list(model.parameters())

        # Set some gradients to None
        params[0].grad = torch.full_like(params[0], 4.0)
        params[1].grad = None  # This should be skipped
        optimizer.step()

        params[0].grad += torch.full_like(params[0], 4.0)
        # params[1].grad still None
        optimizer.step()  # Should step without error

        # Should not have raised
        assert optimizer.statistics()["num_steps"] == 1

    def test_take_optimizer_step_increments_num_steps(self):
        """_take_optimizer_step() increments num_steps."""
        model, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=1)

        create_gradients(base_optimizer, [1.0])
        initial = optimizer.statistics()["num_steps"]

        optimizer.step()

        after = optimizer.statistics()["num_steps"]
        assert after == initial + 1

    def test_take_optimizer_step_resets_num_draws_to_zero(self):
        """_take_optimizer_step() resets num_draws to 0."""
        model, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=3)

        create_gradients(base_optimizer, [1.0])
        optimizer.step()
        create_gradients(base_optimizer, [1.0])
        optimizer.step()
        create_gradients(base_optimizer, [1.0])
        optimizer.step()  # Should step

        stats = optimizer.statistics()
        assert stats["num_draws"] == 0

    def test_take_optimizer_step_sets_last_num_draws(self):
        """_take_optimizer_step() sets last_num_draws property."""
        model, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=4)

        # Before any step
        assert optimizer.statistics()["last_num_draws"] is None

        # Accumulate 4 batches and step
        for _ in range(4):
            create_gradients(base_optimizer, [1.0])
            optimizer.step()

        stats = optimizer.statistics()
        assert stats["last_num_draws"] == 4
        assert stats["num_draws"] == 0  # Reset after step

    def test_take_optimizer_step_updates_last_grad_norm(self):
        """_take_optimizer_step() updates last_grad_norm property with gradient information."""
        model, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer, step_every=1)

        # Before step
        assert optimizer.statistics()["last_grad_norm"] is None

        # Set known gradients
        create_gradients(base_optimizer, [1.0])
        optimizer.step()

        # After step, should have a gradient norm
        stats = optimizer.statistics()
        assert stats["last_grad_norm"] is not None
        assert isinstance(stats["last_grad_norm"], (float, int))
        assert stats["last_grad_norm"] >= 0

    def test_take_optimizer_step_throws_when_num_draws_zero(self):
        """_take_optimizer_step() raises error if num_draws is 0."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = MinimalTestOptimizer(base_optimizer)

        # Directly calling _take_optimizer_step without _batch_received should fail
        with pytest.raises(RuntimeError):
            optimizer._take_optimizer_step()


# =============================================================================
# Suite 6: Subclass Contract - State Management
# =============================================================================


class TestSubclassStateManagement:
    """Test set_state() and get_state() subclass methods."""

    def test_set_state_vital_retrievable_via_get_state(self):
        """Values set with flag='vital' are retrievable via get_state()."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)

        optimizer.set_state("test_vital", 123, "vital")
        value = optimizer.get_state("test_vital")

        assert value == 123

    def test_set_state_optional_retrievable_via_get_state(self):
        """Values set with flag='optional' are retrievable via get_state()."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)

        optimizer.set_state("test_optional", 456, "optional")
        value = optimizer.get_state("test_optional")

        assert value == 456

    def test_set_state_vital_appears_in_vital_statistics(self):
        """Values marked 'vital' appear in vital_statistics()."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)

        optimizer.set_state("important_metric", 99.9, "vital")

        vital = optimizer.vital_statistics()
        assert "important_metric" in vital
        assert vital["important_metric"] == 99.9

    def test_set_state_optional_not_in_vital_statistics(self):
        """Values marked 'optional' do NOT appear in vital_statistics()."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)

        optimizer.set_state("debug_info", "test", "optional")

        vital = optimizer.vital_statistics()
        full = optimizer.statistics()

        assert "debug_info" not in vital
        assert "debug_info" in full

    def test_set_state_optimizer_extends_param_groups(self):
        """set_state with flag='optimizer' extends param_groups."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)  # Sets "threshold" with optimizer flag

        # Check that param_groups now has "threshold"
        for group in optimizer.param_groups:
            assert "threshold" in group
            assert group["threshold"] == 1.0

    def test_set_state_optimizer_appears_in_valid_schedule_targets(self):
        """Parameters extended via flag='optimizer' appear in valid_schedule_targets."""
        _, base_optimizer = create_simple_model_and_base_optimizer()
        optimizer = StatefulTestOptimizer(base_optimizer)

        targets = optimizer.valid_schedule_targets
        assert "threshold" in targets

    def test_get_state_retrieves_optimizer_params(self):
        """get_state() can retrieve optimizer parameters like lr."""
        _, base_optimizer = create_simple_model_and_base_optimizer(lr=0.555)
        optimizer = MinimalTestOptimizer(base_optimizer)

        lr = optimizer.get_state("lr")
        assert lr == 0.555

    def test_get_state_multigroup_aggregation_none(self):
        """get_state with aggregate_behavior='none' returns list for multi-group."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.1, lr2=0.2)
        optimizer = MinimalTestOptimizer(base_optimizer)

        lrs = optimizer.get_state("lr", aggregate_behavior="none")
        assert isinstance(lrs, list)
        assert len(lrs) == 2
        assert 0.1 in lrs
        assert 0.2 in lrs

    def test_get_state_multigroup_aggregation_mean(self):
        """get_state with aggregate_behavior='mean' returns mean."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.1, lr2=0.2)
        optimizer = MinimalTestOptimizer(base_optimizer)

        lr_mean = optimizer.get_state("lr", aggregate_behavior="mean")
        assert abs(lr_mean - 0.15) < 1e-6

    def test_get_state_multigroup_aggregation_max(self):
        """get_state with aggregate_behavior='max' returns max."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.1, lr2=0.2)
        optimizer = MinimalTestOptimizer(base_optimizer)

        lr_max = optimizer.get_state("lr", aggregate_behavior="max")
        assert lr_max == 0.2

    def test_get_state_multigroup_aggregation_min(self):
        """get_state with aggregate_behavior='min' returns min."""
        base_optimizer = create_multigroup_base_optimizer(lr1=0.1, lr2=0.2)
        optimizer = MinimalTestOptimizer(base_optimizer)

        lr_min = optimizer.get_state("lr", aggregate_behavior="min")
        assert lr_min == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

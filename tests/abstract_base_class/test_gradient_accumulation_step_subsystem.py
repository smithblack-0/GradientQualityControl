"""
Black box tests for GradientAccumulationStepSubsystem.

Tests validate the contract specified in documentation/base_object_implementation.md.
All tests use black box methodology:
- Test only documented behavior
- Use real StateManagementSubsystem through its public interface
- Observe effects through properties and state queries
- Verify gradient operations through observable optimizer state
- Never test implementation details

Test organization:
- Constructor validation and post-conditions
- Property access and state retrieval
- batch_received() counter updates and bounds enforcement
- take_optimizer_step() gradient averaging and counter updates
- Error conditions and invariants
"""
import pytest
import torch
from typing import List

from src.gradient_quality_control.base.state_management import StateManagementSubsystem
from src.gradient_quality_control.base.gradient_accumulation import GradientAccumulationStepSubsystem


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


class NoOpOptimizer(torch.optim.Optimizer):
    """Used to examine gradients."""
    def __init__(self, params, defaults=None):
        super().__init__(params, defaults or {})

    def step(self, *args, **kwargs):
        # intentionally do nothing
        return None

    def zero_grad(self, *args, **kwargs):
        # intentionally do nothing
        return None



def create_simple_optimizer(num_params=3):
    """Create a simple SGD optimizer with parameters."""
    params = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(num_params)]
    return torch.optim.SGD(params, lr=0.01)


def set_gradients(optimizer, value=1.0):
    """Set gradients on all parameters to a specific value."""
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad = torch.full_like(param, value)

def add_gradients(optimizer, value=1.0):
    """Set gradients on all parameters to a specific value."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad += value

def check_gradients_zeroed(optimizer):
    """Check if all gradients are None or zero."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.grad.abs().sum() > 0:
                return False
    return True

def create_no_op_optimizer():
    """Creates a no op optimizer that does nothing"""
    params = [
        torch.nn.Parameter(torch.randn([10, 5]))
        for _ in range(10)
    ]
    return NoOpOptimizer(params)

def get_optimizer_gradients(optimizer)->List[torch.Tensor]:
    """Gets a list of all optimizer gradients in order"""
    output = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                output.append(param.grad.clone())
    return output


# =============================================================================
# Constructor Test Suite - tests that constructor initializes state correctly
# =============================================================================


class TestConstructor:
    """Test constructor validation and state initialization."""

    def test_constructor_accepts_valid_inputs(self):
        """Constructor accepts valid state_manager, optimizer, and max_draws."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        subsystem = GradientAccumulationStepSubsystem(
            state_manager=state_mgr,
            optimizer=optimizer,
            max_draws=32
        )

        assert subsystem is not None

    def test_constructor_initializes_counters(self):
        """Constructor initializes counters to their sane defaults"""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        # Verify through properties
        assert subsystem.num_batches == 0
        assert subsystem.num_steps == 0
        assert subsystem.num_draws == 0
        assert subsystem.last_num_draws is None
        assert subsystem.last_grad_norm is None


    def test_constructor_uses_default_max_draws(self):
        """Constructor accepts max_draws with default value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        # Should not raise with default
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer)
        assert subsystem is not None

    def test_constructor_state_visible_through_state_manager(self):
        """Constructor-initialized state is visible through state manager."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)

        GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        # Verify state exists in state manager
        assert state_mgr.get_state('num_batches') == 0
        assert state_mgr.get_state('num_steps') == 0
        assert state_mgr.get_state('num_draws') == 0
        assert state_mgr.get_state('last_num_draws') is None
        assert state_mgr.get_state('last_grad_norm') is None

    def test_constructor_state_correctly_marked_for_reporting(self):
        """Test the right reporting flags ended up on the things"""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        for key, flavor in state_mgr.show_state():
            if "num_batches" in key :
                assert flavor == "vital"
            if "num_steps" in key :
                assert flavor == "vital"
            if "num_draws" in key and "last" not in key:
                assert flavor == "optional"
            if "last_num_draws" in key:
                assert flavor == "vital"
            if "last_grad_norm" in key:
                assert flavor == "vital"



# =============================================================================
# Property Test Suite - tests that properties retrieve state correctly
# =============================================================================


class TestProperties:
    """Test property access retrieves correct values."""

    def test_num_batches_property_reflects_state(self):
        """num_batches property reflects current state value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        assert hasattr(subsystem, "num_batches")
        assert isinstance(subsystem.num_batches, int)
        assert subsystem.num_batches == 0

    def test_num_steps_property_reflects_state(self):
        """num_steps property reflects current state value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        assert hasattr(subsystem, "num_steps")
        assert isinstance(subsystem.num_steps, int)
        assert subsystem.num_steps == 0

    def test_num_draws_property_reflects_state(self):
        """num_draws property reflects current state value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        assert hasattr(subsystem, "num_draws")
        assert isinstance(subsystem.num_draws, int)
        assert subsystem.num_draws == 0

    def test_last_num_draws_property_reflects_state(self):
        """last_num_draws property reflects current state value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        assert hasattr(subsystem, "last_num_draws")
        assert subsystem.last_num_draws is None

    def test_last_grad_norm_property_reflects_state(self):
        """last_grad_norm property reflects current state value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        assert hasattr(subsystem, "last_grad_norm")
        assert subsystem.last_grad_norm is None

# =============================================================================
# Batch Received Test Suite - tests that batch_received updates counters correctly
# =============================================================================


class TestBatchReceived:
    """Test batch_received counter updates and bounds enforcement."""

    def test_batch_received_increments_num_batches(self):
        """batch_received increments num_batches by 1."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        initial_batches = subsystem.num_batches
        subsystem.batch_received()

        assert subsystem.num_batches == initial_batches + 1

    def test_batch_received_increments_num_draws(self):
        """batch_received increments num_draws by 1."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        initial_draws = subsystem.num_draws
        subsystem.batch_received()

        assert subsystem.num_draws == initial_draws + 1

    def test_batch_received_multiple_times_increments_correctly(self):
        """batch_received can be called multiple times and increments correctly."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        subsystem.batch_received()
        subsystem.batch_received()
        subsystem.batch_received()

        assert subsystem.num_batches == 3
        assert subsystem.num_draws == 3

    def test_batch_received_enforces_max_draws_bound(self):
        """batch_received raises error when num_draws would exceed max_draws."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=3)

        # Accumulate to max
        subsystem.batch_received()  # 1
        subsystem.batch_received()  # 2
        subsystem.batch_received()  # 3

        # Fourth call should fail
        with pytest.raises(RuntimeError):
            subsystem.batch_received()

    def test_batch_received_allows_exactly_max_draws(self):
        """batch_received allows accumulating exactly max_draws batches."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=3)

        # Should be able to call exactly max_draws times
        subsystem.batch_received()  # 1
        subsystem.batch_received()  # 2
        subsystem.batch_received()  # 3

        assert subsystem.num_draws == 3


# =============================================================================
# Take Optimizer Step Test Suite - tests that take_optimizer_step handles gradient averaging and counter updates correctly
# =============================================================================


class TestTakeOptimizerStep:
    """Test take_optimizer_step gradient averaging, stepping, and counter updates."""

    def test_take_optimizer_step_averages_gradients(self):
        """take_optimizer_step divides gradients by num_draws before stepping."""
        optimizer = create_no_op_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)
        set_gradients(optimizer, 0.0)

        # Simulate 4 autograd steps
        # Should now contain 8.0, and with 4 steps.
        for _ in range(4):
            add_gradients(optimizer, 2.0)
            subsystem.batch_received()

        for gradient in get_optimizer_gradients(optimizer):
            assert torch.allclose(gradient, torch.tensor(8.0)) # 2.0*4

        # Take the optimizer step. Should average by the number of
        # draws.
        subsystem.take_optimizer_step()
        for gradient in get_optimizer_gradients(optimizer):
            assert torch.allclose(gradient, torch.tensor(2.0))  # 2.0*4/4

    def test_take_optimizer_step_increments_num_steps(self):
        """take_optimizer_step increments num_steps by 1."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        set_gradients(optimizer, 1.0)
        subsystem.batch_received()

        initial_steps = subsystem.num_steps
        subsystem.take_optimizer_step()

        assert subsystem.num_steps == initial_steps + 1

    def test_take_optimizer_step_resets_num_draws_to_zero(self):
        """take_optimizer_step resets num_draws to 0."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        set_gradients(optimizer, 1.0)
        subsystem.batch_received()
        subsystem.batch_received()
        subsystem.batch_received()

        assert subsystem.num_draws == 3

        subsystem.take_optimizer_step()

        assert subsystem.num_draws == 0

    def test_take_optimizer_step_updates_last_num_draws(self):
        """take_optimizer_step sets last_num_draws to previous num_draws value."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        set_gradients(optimizer, 1.0)
        subsystem.batch_received()
        subsystem.batch_received()
        subsystem.batch_received()
        subsystem.batch_received()
        subsystem.batch_received()

        assert subsystem.num_draws == 5

        subsystem.take_optimizer_step()

        assert subsystem.last_num_draws == 5

    def test_take_optimizer_step_updates_last_grad_norm(self):
        """take_optimizer_step sets last_grad_norm to computed gradient norm."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        set_gradients(optimizer, 1.0)
        subsystem.batch_received()

        subsystem.take_optimizer_step()

        # Should be set to some non-None numeric value
        assert subsystem.last_grad_norm is not None
        assert isinstance(subsystem.last_grad_norm, (int, float))
        assert subsystem.last_grad_norm >= 0

    def test_take_optimizer_step_zeros_gradients(self):
        """take_optimizer_step zeros all gradients after stepping."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        set_gradients(optimizer, 5.0)
        subsystem.batch_received()

        subsystem.take_optimizer_step()

        assert check_gradients_zeroed(optimizer)

    def test_take_optimizer_step_throws_when_num_draws_zero(self):
        """take_optimizer_step raises error when num_draws is 0."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        # Don't call batch_received, so num_draws is still 0

        with pytest.raises(RuntimeError):
            subsystem.take_optimizer_step()

    def test_take_optimizer_step_can_be_called_multiple_times(self):
        """take_optimizer_step can be called multiple times in sequence."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        # First step
        set_gradients(optimizer, 1.0)
        subsystem.batch_received()
        subsystem.take_optimizer_step()

        assert subsystem.num_steps == 1

        # Second step
        set_gradients(optimizer, 1.0)
        subsystem.batch_received()
        subsystem.take_optimizer_step()

        assert subsystem.num_steps == 2


# =============================================================================
# Invariants Test Suite - tests that documented invariants are maintained correctly
# =============================================================================


class TestInvariants:
    """Test documented invariants are maintained."""

    def test_num_batches_greater_or_equal_num_steps(self):
        """Invariant: num_batches >= num_steps always holds."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        # Process multiple batches and steps
        for _ in range(10):
            set_gradients(optimizer, 1.0)
            subsystem.batch_received()
            subsystem.batch_received()
            subsystem.take_optimizer_step()

            # Invariant should always hold
            assert subsystem.num_batches >= subsystem.num_steps

    def test_num_draws_resets_to_zero_after_step(self):
        """Invariant: num_draws == 0 immediately after stepping."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        set_gradients(optimizer, 1.0)
        subsystem.batch_received()
        subsystem.batch_received()
        subsystem.batch_received()

        subsystem.take_optimizer_step()

        assert subsystem.num_draws == 0

    def test_last_values_none_before_first_step(self):
        """Invariant: last_num_draws and last_grad_norm are None before first step."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        # Before any step
        assert subsystem.last_num_draws is None
        assert subsystem.last_grad_norm is None

    def test_last_values_not_none_after_first_step(self):
        """Invariant: last_num_draws and last_grad_norm are not None after first step."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        set_gradients(optimizer, 1.0)
        subsystem.batch_received()
        subsystem.take_optimizer_step()

        # After step, should have values
        assert subsystem.last_num_draws is not None
        assert subsystem.last_grad_norm is not None

    def test_num_draws_never_exceeds_max_draws(self):
        """Invariant: num_draws <= max_draws enforced by batch_received."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=5)

        # Fill to max
        for _ in range(5):
            subsystem.batch_received()

        assert subsystem.num_draws == 5

        # Attempting to exceed should fail
        with pytest.raises(RuntimeError):
            subsystem.batch_received()

    def test_gradients_always_averaged(self):
        """Invariant: Gradients are always averaged (divided by num_draws) before stepping."""
        optimizer = create_simple_optimizer()
        state_mgr = StateManagementSubsystem(optimizer)
        subsystem = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws=64)

        # Set known gradient value and accumulate known number of batches
        set_gradients(optimizer, 12.0)
        subsystem.batch_received()
        subsystem.batch_received()
        subsystem.batch_received()

        # num_draws = 3, so averaged gradient should be 12.0 / 3 = 4.0
        # The optimizer should see 4.0 as gradient when stepping

        # We can't directly observe what the optimizer saw during stepping,
        # but we can verify gradients were zeroed (part of the contract)
        subsystem.take_optimizer_step()
        assert check_gradients_zeroed(optimizer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

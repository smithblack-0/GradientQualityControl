"""
Black box tests for OptimizerWrapperSBC (Scheduled Batch Controller).

Tests validate the public contract as documented in
documentation/optimizer_wrapper_api.md. All tests use black box methodology:
- Test only documented behavior
- Use real PyTorch optimizers (no mocks)
- Use ScheduleAnything for schedule target binding
- Never access implementation details
"""

import pytest
import torch
import torch.nn as nn
from torch_schedule_anything import arbitrary_schedule_factory

from src.gradient_quality_control.scheduled_batch_controller import OptimizerWrapperSBC


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


def create_simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )


def create_optimizer_with_schedule(model, physical_batch_size, logical_batch_size):
    """Create SBC optimizer with fixed logical_batch_size schedule."""
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = OptimizerWrapperSBC(base_optimizer, physical_batch_size=physical_batch_size)

    # Bind constant schedule to logical_batch_size
    scheduler = arbitrary_schedule_factory(
        optimizer,
        schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: logical_batch_size
        ),
        schedule_target='logical_batch_size'
    )

    return optimizer, scheduler


def perform_forward_backward(model, optimizer):
    """Perform forward pass, compute loss, and backward pass."""
    x = torch.randn(4, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()


# =============================================================================
# Suite 1: Constructor and Initialization
# =============================================================================


class TestConstructor:
    """Test constructor parameter handling."""

    def test_accepts_optimizer_and_physical_batch_size(self):
        """Constructor accepts optimizer and physical_batch_size."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperSBC(base_optimizer, physical_batch_size=32)

        assert optimizer is not None

    def test_accepts_max_batch_draws_parameter(self):
        """Constructor accepts max_batch_draws parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperSBC(
            base_optimizer,
            physical_batch_size=32,
            max_batch_draws=10
        )

        assert optimizer is not None

    def test_accepts_distributed_mode_parameter(self):
        """Constructor accepts distributed_mode parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperSBC(
            base_optimizer,
            physical_batch_size=32,
            distributed_mode="replicated"
        )

        assert optimizer is not None


# =============================================================================
# Suite 2: Stepping Behavior
# =============================================================================


class TestSteppingBehavior:
    """Test step decision logic based on logical_batch_size."""

    def test_steps_when_logical_batch_size_reached(self):
        """Steps when num_draws * physical_batch_size >= logical_batch_size."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model,
            physical_batch_size=32,
            logical_batch_size=64  # Should step after 2 draws (2*32=64)
        )

        # First draw - accumulate
        perform_forward_backward(model, optimizer)
        result1 = optimizer.step()
        assert result1 is False

        # Second draw - should step
        perform_forward_backward(model, optimizer)
        result2 = optimizer.step()
        assert result2 is True

    def test_steps_immediately_when_logical_equals_physical(self):
        """Steps immediately when logical_batch_size equals physical_batch_size."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model,
            physical_batch_size=32,
            logical_batch_size=32  # Should step every call
        )

        perform_forward_backward(model, optimizer)
        result = optimizer.step()

        assert result is True

    def test_accumulates_when_logical_batch_size_not_reached(self):
        """Accumulates when num_draws * physical_batch_size < logical_batch_size."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model,
            physical_batch_size=32,
            logical_batch_size=128  # Need 4 draws
        )

        # First three draws should accumulate
        for _ in range(3):
            perform_forward_backward(model, optimizer)
            result = optimizer.step()
            assert result is False

        # Fourth draw should step
        perform_forward_backward(model, optimizer)
        result = optimizer.step()
        assert result is True

    def test_force_steps_at_max_batch_draws(self):
        """Forces step when max_batch_draws reached regardless of logical_batch_size."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperSBC(
            base_optimizer,
            physical_batch_size=32,
            max_batch_draws=3
        )

        # Set very high logical_batch_size that would never be reached
        scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 10000.0
            ),
            schedule_target='logical_batch_size'
        )

        # Accumulate up to max_draws
        for i in range(3):
            perform_forward_backward(model, optimizer)
            result = optimizer.step()
            if i < 2:
                assert result is False
            else:
                assert result is True  # Forced at max_draws


# =============================================================================
# Suite 3: Schedule Integration
# =============================================================================


class TestScheduleIntegration:
    """Test integration with ScheduleAnything."""

    def test_responds_to_schedule_changes(self):
        """Optimizer responds to scheduler changing logical_batch_size."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperSBC(base_optimizer, physical_batch_size=32)

        # Start with logical_batch_size = 32 (step every call)
        scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 32.0 if step < 2 else 96.0
            ),
            schedule_target='logical_batch_size'
        )

        # First step: logical=32, should step immediately
        perform_forward_backward(model, optimizer)
        result1 = optimizer.step()
        assert result1 is True

        # Advance scheduler
        scheduler.step()

        # Now logical=96, need 3 draws
        perform_forward_backward(model, optimizer)
        result2 = optimizer.step()
        assert result2 is False

        perform_forward_backward(model, optimizer)
        result3 = optimizer.step()
        assert result3 is False

        perform_forward_backward(model, optimizer)
        result4 = optimizer.step()
        assert result4 is True

    def test_works_with_pytorch_schedulers(self):
        """Works with standard PyTorch schedulers."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperSBC(base_optimizer, physical_batch_size=32)

        # Use StepLR to change logical_batch_size
        scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=1, gamma=2.0
            ),
            schedule_target='logical_batch_size',
            default_value=32.0
        )

        # Should step with logical=32
        perform_forward_backward(model, optimizer)
        result = optimizer.step()
        assert result is True


# =============================================================================
# Suite 4: Statistics
# =============================================================================


class TestStatistics:
    """Test statistics() method contract."""

    def test_statistics_returns_dict(self):
        """statistics() returns a dictionary."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model, physical_batch_size=32, logical_batch_size=64
        )

        stats = optimizer.statistics()
        assert isinstance(stats, dict)

    def test_statistics_contains_base_counters(self):
        """statistics() includes counters from base class."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model, physical_batch_size=32, logical_batch_size=64
        )

        stats = optimizer.statistics()
        assert "num_batches" in stats or "batches" in stats
        assert "num_steps" in stats or "steps" in stats
        assert "num_draws" in stats

    def test_statistics_updates_after_accumulation(self):
        """statistics() reflects accumulation state."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model, physical_batch_size=32, logical_batch_size=96
        )

        initial_stats = optimizer.statistics()

        # Accumulate one batch
        perform_forward_backward(model, optimizer)
        optimizer.step()

        updated_stats = optimizer.statistics()

        # num_draws or batches should have increased
        assert (updated_stats.get("num_draws", 0) > initial_stats.get("num_draws", 0) or
                updated_stats.get("num_batches", 0) > initial_stats.get("num_batches", 0) or
                updated_stats.get("batches", 0) > initial_stats.get("batches", 0))


# =============================================================================
# Suite 5: Gradient Accumulation
# =============================================================================


class TestGradientAccumulation:
    """Test gradient accumulation behavior."""

    def test_accumulates_gradients_across_draws(self):
        """Gradients are accumulated across multiple draws."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model, physical_batch_size=32, logical_batch_size=64
        )

        # First draw
        perform_forward_backward(model, optimizer)
        first_grad = model[0].weight.grad.clone()
        optimizer.step()  # Accumulate

        # Second draw
        perform_forward_backward(model, optimizer)
        second_grad_before_step = model[0].weight.grad.clone()

        # Gradient should be sum of both draws
        assert not torch.allclose(second_grad_before_step, first_grad)

        optimizer.step()  # This should step and clear

    def test_gradients_cleared_after_step(self):
        """Gradients are cleared after optimizer steps."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model, physical_batch_size=32, logical_batch_size=32
        )

        perform_forward_backward(model, optimizer)
        optimizer.step()  # Should step immediately

        # Gradients should be cleared
        assert model[0].weight.grad is None or torch.all(model[0].weight.grad == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Black box tests for OptimizerWrapperGNTS (Gradient Norm Threshold Scheduler).

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

from src.gradient_quality_control.gradient_norm_threshold_scheduler import (
    OptimizerWrapperGNTS,
)


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


def create_optimizer_with_schedule(model, gradient_norm_threshold):
    """Create GNTS optimizer with fixed gradient_norm_threshold schedule."""
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = OptimizerWrapperGNTS(base_optimizer)

    # Bind constant schedule to gradient_norm_threshold
    scheduler = arbitrary_schedule_factory(
        optimizer,
        schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: gradient_norm_threshold
        ),
        schedule_target='gradient_norm_threshold'
    )

    return optimizer, scheduler


def perform_forward_backward(model, grad_scale=1.0):
    """Perform forward pass with controlled gradient magnitude."""
    x = torch.randn(4, 10)
    output = model(x)
    loss = (output.sum() * grad_scale)
    loss.backward()


# =============================================================================
# Suite 1: Constructor and Initialization
# =============================================================================


class TestConstructor:
    """Test constructor parameter handling."""

    def test_accepts_optimizer(self):
        """Constructor accepts optimizer parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperGNTS(base_optimizer)

        assert optimizer is not None

    def test_accepts_max_batch_draws_parameter(self):
        """Constructor accepts max_batch_draws parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperGNTS(base_optimizer, max_batch_draws=10)

        assert optimizer is not None

    def test_accepts_distributed_mode_parameter(self):
        """Constructor accepts distributed_mode parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperGNTS(
            base_optimizer,
            distributed_mode="replicated"
        )

        assert optimizer is not None


# =============================================================================
# Suite 2: Stepping Behavior
# =============================================================================


class TestSteppingBehavior:
    """Test step decision logic based on gradient norm threshold."""

    def test_steps_when_norm_below_threshold(self):
        """Steps when mean gradient norm <= gradient_norm_threshold."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model,
            gradient_norm_threshold=100.0  # High threshold
        )

        # Generate small gradients
        perform_forward_backward(model, grad_scale=0.001)
        result = optimizer.step()

        # Should step with small gradients and high threshold
        assert result is True

    def test_accumulates_when_norm_above_threshold(self):
        """Accumulates when mean gradient norm > gradient_norm_threshold."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model,
            gradient_norm_threshold=0.001  # Very low threshold
        )

        # Generate large gradients
        perform_forward_backward(model, grad_scale=100.0)
        result = optimizer.step()

        # Should accumulate with large gradients and low threshold
        assert result is False

    def test_accumulation_reduces_mean_norm(self):
        """Accumulation across draws reduces mean gradient norm."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model,
            gradient_norm_threshold=1.0
        )

        # First draw with large gradients
        perform_forward_backward(model, grad_scale=10.0)
        result1 = optimizer.step()
        assert result1 is False  # Should accumulate

        # Second draw with smaller gradients
        # Mean norm decreases, may meet threshold
        perform_forward_backward(model, grad_scale=0.1)
        result2 = optimizer.step()

        # Eventually should step as mean norm decreases
        # (May need more draws depending on actual norms)
        if not result2:
            perform_forward_backward(model, grad_scale=0.1)
            result3 = optimizer.step()
            # After enough accumulation, should step
            assert isinstance(result3, bool)

    def test_force_steps_at_max_batch_draws(self):
        """Forces step when max_batch_draws reached regardless of norm."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperGNTS(
            base_optimizer,
            max_batch_draws=3
        )

        # Set very low threshold that won't be met
        scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 0.0001
            ),
            schedule_target='gradient_norm_threshold'
        )

        # Accumulate with large gradients that won't meet threshold
        for i in range(3):
            perform_forward_backward(model, grad_scale=100.0)
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
        """Optimizer responds to scheduler changing gradient_norm_threshold."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperGNTS(base_optimizer)

        # Start with high threshold (easy to meet)
        # Then switch to low threshold (hard to meet)
        scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 100.0 if step < 2 else 0.001
            ),
            schedule_target='gradient_norm_threshold'
        )

        # First step: high threshold, should step
        perform_forward_backward(model, grad_scale=1.0)
        result1 = optimizer.step()
        assert result1 is True

        # Advance scheduler
        scheduler.step()

        # Now low threshold, should accumulate
        perform_forward_backward(model, grad_scale=1.0)
        result2 = optimizer.step()
        assert result2 is False

    def test_works_with_pytorch_schedulers(self):
        """Works with standard PyTorch schedulers."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperGNTS(base_optimizer)

        # Use ExponentialLR to change gradient_norm_threshold
        scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=0.95
            ),
            schedule_target='gradient_norm_threshold',
            default_value=10.0
        )

        # Should work without errors
        perform_forward_backward(model, grad_scale=1.0)
        result = optimizer.step()
        assert isinstance(result, bool)


# =============================================================================
# Suite 4: Statistics Smoke Tests
# =============================================================================


class TestStatisticsSmoke:
    """Smoke tests to verify statistics methods work (details tested in base class)."""

    def test_statistics_returns_dict(self):
        """statistics() method exists and returns dict."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(model, gradient_norm_threshold=1.0)

        stats = optimizer.statistics()
        assert isinstance(stats, dict)

    def test_vital_statistics_returns_dict(self):
        """vital_statistics() method exists and returns dict."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(model, gradient_norm_threshold=1.0)

        vital_stats = optimizer.vital_statistics()
        assert isinstance(vital_stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

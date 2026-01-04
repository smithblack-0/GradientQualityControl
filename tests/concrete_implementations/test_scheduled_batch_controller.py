"""
Black box tests for OptimizerWrapperSBC (Scheduled Batch Controller).

Tests validate the public contract as documented in
documentation/optimizer_wrapper_api.md. All tests use black box methodology:
- Test only documented behavior
- Use real PyTorch optimizers (no mocks)
- Use ScheduleAnything for schedule target binding
- Never access implementation details
"""

import math
import pytest
import torch
import torch.nn as nn
from torch_schedule_anything import arbitrary_schedule_factory, SynchronousSchedule

from src.gradient_quality_control.scheduled_batch_controller import (
    OptimizerWrapperSBC,
    make_sbc_with_polynomial_schedule,
    make_sbc_with_polynomial_schedule_conventional_lr,
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
# Suite 4: Statistics Smoke Tests
# =============================================================================


class TestStatisticsSmoke:
    """Smoke tests to verify statistics methods work (details tested in base class)."""

    def test_statistics_returns_dict(self):
        """statistics() method exists and returns dict."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model, physical_batch_size=32, logical_batch_size=64
        )

        stats = optimizer.statistics()
        assert isinstance(stats, dict)

    def test_vital_statistics_returns_dict(self):
        """vital_statistics() method exists and returns dict."""
        model = create_simple_model()
        optimizer, scheduler = create_optimizer_with_schedule(
            model, physical_batch_size=32, logical_batch_size=64
        )

        vital_stats = optimizer.vital_statistics()
        assert isinstance(vital_stats, dict)


# =============================================================================
# Suite 5: Factory Tests
# =============================================================================


class TestMakeSBCWithPolynomialSchedule:
    """Test make_sbc_with_polynomial_schedule factory."""

    def test_factory_exists_and_callable(self):
        """Factory function exists and is callable."""
        assert callable(make_sbc_with_polynomial_schedule)

    def test_returns_correct_types(self):
        """Factory returns (OptimizerWrapperSBC, scheduler) tuple."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

        optimizer, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=base_optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        assert isinstance(optimizer, OptimizerWrapperSBC)
        assert isinstance(scheduler, SynchronousSchedule)

    def test_lr_constant_after_warmup(self):
        """Learning rate stays constant after warmup."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
        initial_lr = 0.01

        optimizer, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=base_optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # Step to end of warmup
        for _ in range(100):
            scheduler.step()
        lr_at_100 = scheduler.get_last_lr()[0]

        # Step further into training
        for _ in range(400):
            scheduler.step()
        lr_at_500 = scheduler.get_last_lr()[0]

        # LR should be constant at initial_lr
        assert math.isclose(lr_at_100, initial_lr, rel_tol=0.01)
        assert math.isclose(lr_at_500, initial_lr, rel_tol=0.01)

    def test_batch_size_polynomial_schedule(self):
        """Batch size follows polynomial schedule from initial to final."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

        optimizer, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=base_optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
            polynomial_power=2.0,
        )

        # At end of warmup, batch size should be at initial_batch_size
        for _ in range(100):
            scheduler.step()
        batch_size_at_100 = scheduler.get_last_schedule("logical_batch_size")[0]
        assert math.isclose(batch_size_at_100, 64, rel_tol=0.01)

        # At end of training, should be at final_batch_size
        for _ in range(900):
            scheduler.step()
        batch_size_at_1000 = scheduler.get_last_schedule("logical_batch_size")[0]
        assert math.isclose(batch_size_at_1000, 256, rel_tol=0.01)

    def test_weight_decay_cosine_annealing(self):
        """Weight decay follows cosine annealing to zero after warmup."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
        initial_wd = 0.01

        optimizer, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=base_optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # At end of training, weight decay should anneal to ~0
        for _ in range(1000):
            scheduler.step()
        wd_at_end = scheduler.get_last_schedule("weight_decay")[0]

        # Should be close to zero (annealed down from initial)
        assert wd_at_end < initial_wd * 0.1


class TestMakeSBCWithPolynomialScheduleConventionalLR:
    """Test make_sbc_with_polynomial_schedule_conventional_lr factory."""

    def test_factory_exists_and_callable(self):
        """Factory function exists and is callable."""
        assert callable(make_sbc_with_polynomial_schedule_conventional_lr)

    def test_returns_correct_types(self):
        """Factory returns (OptimizerWrapperSBC, scheduler) tuple."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=base_optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        assert isinstance(optimizer, OptimizerWrapperSBC)
        assert isinstance(scheduler, SynchronousSchedule)

    def test_lr_anneals_after_warmup(self):
        """Learning rate anneals to zero after warmup."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        initial_lr = 0.01

        optimizer, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=base_optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # At end of warmup, LR should be at initial value
        for _ in range(100):
            scheduler.step()
        lr_at_100 = scheduler.get_last_lr()[0]
        assert math.isclose(lr_at_100, initial_lr, rel_tol=0.01)

        # At end of training, LR should anneal toward zero
        for _ in range(900):
            scheduler.step()
        lr_at_1000 = scheduler.get_last_lr()[0]

        # Should be much smaller than initial (annealed down)
        assert lr_at_1000 < initial_lr * 0.1

    def test_batch_size_polynomial_schedule(self):
        """Batch size follows polynomial schedule from initial to final."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=base_optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        batch_size_at_100 = scheduler.get_last_schedule("logical_batch_size")[0]
        assert math.isclose(batch_size_at_100, 64, rel_tol=0.01)

        # At end of training
        for _ in range(900):
            scheduler.step()
        batch_size_at_1000 = scheduler.get_last_schedule("logical_batch_size")[0]
        assert math.isclose(batch_size_at_1000, 256, rel_tol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

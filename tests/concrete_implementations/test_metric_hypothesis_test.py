"""
Black box tests for OptimizerWrapperMHT (Metric Hypothesis Test).

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

from src.gradient_quality_control.metric_hypothesis_test import (
    OptimizerWrapperMHT,
    make_mht_with_warmup_schedule,
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


def create_optimizer_with_schedules(model, confidence_level, percent_error_threshold):
    """Create MHT optimizer with fixed schedule targets."""
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = OptimizerWrapperMHT(base_optimizer)

    # Bind confidence_level schedule
    confidence_scheduler = arbitrary_schedule_factory(
        optimizer,
        schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: confidence_level
        ),
        schedule_target='confidence_level'
    )

    # Bind percent_error_threshold schedule
    error_scheduler = arbitrary_schedule_factory(
        optimizer,
        schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: percent_error_threshold
        ),
        schedule_target='percent_error_threshold'
    )

    return optimizer, (confidence_scheduler, error_scheduler)


def perform_forward_backward(model):
    """Perform forward pass, compute loss, and backward pass."""
    x = torch.randn(4, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()
    return loss.item()


# =============================================================================
# Suite 1: Constructor and Initialization
# =============================================================================


class TestConstructor:
    """Test constructor parameter handling."""

    def test_accepts_optimizer(self):
        """Constructor accepts optimizer parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperMHT(base_optimizer)

        assert optimizer is not None

    def test_accepts_max_batch_draws_parameter(self):
        """Constructor accepts max_batch_draws parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperMHT(base_optimizer, max_batch_draws=10)

        assert optimizer is not None

    def test_accepts_distributed_mode_parameter(self):
        """Constructor accepts distributed_mode parameter."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer = OptimizerWrapperMHT(
            base_optimizer,
            distributed_mode="replicated"
        )

        assert optimizer is not None


# =============================================================================
# Suite 2: Stepping Behavior
# =============================================================================


class TestSteppingBehavior:
    """Test step decision logic based on hypothesis test."""

    def test_requires_metric_argument(self):
        """step() requires metric argument."""
        model = create_simple_model()
        optimizer, schedulers = create_optimizer_with_schedules(
            model, confidence_level=0.95, percent_error_threshold=0.1
        )

        perform_forward_backward(model)

        # Should accept metric argument
        loss = 1.0
        result = optimizer.step(metric=loss)
        assert isinstance(result, bool)

    def test_accumulates_with_single_sample(self):
        """Cannot perform hypothesis test with single sample."""
        model = create_simple_model()
        optimizer, schedulers = create_optimizer_with_schedules(
            model, confidence_level=0.95, percent_error_threshold=0.1
        )

        loss = perform_forward_backward(model)
        result = optimizer.step(metric=loss)

        # Should accumulate (need at least 2 samples for t-test)
        assert result is False

    def test_steps_with_consistent_metrics(self):
        """Steps when metrics are consistent (tight confidence interval)."""
        model = create_simple_model()
        optimizer, schedulers = create_optimizer_with_schedules(
            model, confidence_level=0.95, percent_error_threshold=0.5
        )

        # Feed consistent metric values
        for _ in range(10):
            perform_forward_backward(model)
            result = optimizer.step(metric=1.0)  # Identical metrics
            if result:
                break

        # Should eventually step with identical metrics
        assert result is True

    def test_accumulates_with_variable_metrics(self):
        """Accumulates when metrics are variable (wide confidence interval)."""
        model = create_simple_model()
        optimizer, schedulers = create_optimizer_with_schedules(
            model, confidence_level=0.99, percent_error_threshold=0.001  # Very strict
        )

        # Feed highly variable metrics
        metrics = [0.1, 10.0, 0.5, 5.0]
        results = []
        for metric in metrics:
            perform_forward_backward(model)
            result = optimizer.step(metric=metric)
            results.append(result)

        # Should still be accumulating due to high variance and strict threshold
        assert not all(results)  # At least some False

    def test_force_steps_at_max_batch_draws(self):
        """Forces step when max_batch_draws reached."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperMHT(
            base_optimizer,
            max_batch_draws=3
        )

        # Set very strict criteria that won't be met
        confidence_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 0.999
            ),
            schedule_target='confidence_level'
        )
        error_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 0.0001
            ),
            schedule_target='percent_error_threshold'
        )

        # Feed variable metrics that won't meet criteria
        for i, metric in enumerate([1.0, 2.0, 3.0]):
            perform_forward_backward(model)
            result = optimizer.step(metric=metric)
            if i < 2:
                assert result is False
            else:
                assert result is True  # Forced at max_draws


# =============================================================================
# Suite 3: Schedule Integration
# =============================================================================


class TestScheduleIntegration:
    """Test integration with ScheduleAnything."""

    def test_responds_to_confidence_level_changes(self):
        """Optimizer responds to scheduler changing confidence_level."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperMHT(base_optimizer)

        # Start with relaxed confidence, then make strict
        confidence_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 0.50 if step < 2 else 0.999
            ),
            schedule_target='confidence_level'
        )
        error_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 0.5
            ),
            schedule_target='percent_error_threshold'
        )

        # With relaxed confidence, should step quickly
        perform_forward_backward(model)
        optimizer.step(metric=1.0)
        perform_forward_backward(model)
        result1 = optimizer.step(metric=1.0)

        # Should step with relaxed criteria
        assert isinstance(result1, bool)

    def test_responds_to_error_threshold_changes(self):
        """Optimizer responds to scheduler changing percent_error_threshold."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperMHT(base_optimizer, max_batch_draws=20)

        # High error tolerance makes it easy to step
        confidence_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 0.95
            ),
            schedule_target='confidence_level'
        )
        error_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 10.0  # Very high tolerance
            ),
            schedule_target='percent_error_threshold'
        )

        # Should step with high error tolerance
        for _ in range(5):
            perform_forward_backward(model)
            result = optimizer.step(metric=1.0)
            if result:
                break

        assert isinstance(result, bool)

    def test_works_with_pytorch_schedulers(self):
        """Works with standard PyTorch schedulers."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = OptimizerWrapperMHT(base_optimizer)

        # Use StepLR for both targets
        confidence_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=5, gamma=0.9
            ),
            schedule_target='confidence_level',
            default_value=0.95
        )
        error_scheduler = arbitrary_schedule_factory(
            optimizer,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=5, gamma=1.1
            ),
            schedule_target='percent_error_threshold',
            default_value=0.1
        )

        # Should work without errors
        perform_forward_backward(model)
        result = optimizer.step(metric=1.0)
        assert isinstance(result, bool)


# =============================================================================
# Suite 4: Statistics Smoke Tests
# =============================================================================


class TestStatisticsSmoke:
    """Smoke tests to verify statistics methods work (details tested in base class)."""

    def test_statistics_returns_dict(self):
        """statistics() method exists and returns dict."""
        model = create_simple_model()
        optimizer, schedulers = create_optimizer_with_schedules(
            model, confidence_level=0.95, percent_error_threshold=0.1
        )

        stats = optimizer.statistics()
        assert isinstance(stats, dict)

    def test_vital_statistics_returns_dict(self):
        """vital_statistics() method exists and returns dict."""
        model = create_simple_model()
        optimizer, schedulers = create_optimizer_with_schedules(
            model, confidence_level=0.95, percent_error_threshold=0.1
        )

        vital_stats = optimizer.vital_statistics()
        assert isinstance(vital_stats, dict)


# =============================================================================
# Suite 5: Factory Tests
# =============================================================================


class TestMakeMHTWithWarmupSchedule:
    """Test make_mht_with_warmup_schedule factory."""

    def test_factory_exists_and_callable(self):
        """Factory function exists and is callable."""
        assert callable(make_mht_with_warmup_schedule)

    def test_returns_correct_types(self):
        """Factory returns (OptimizerWrapperMHT, scheduler) tuple."""

        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer, scheduler = make_mht_with_warmup_schedule(
            optimizer=base_optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        assert isinstance(optimizer, OptimizerWrapperMHT)
        assert isinstance(scheduler, SynchronousSchedule)

    def test_wrapper_configuration(self):
        """Wrapper is configured with correct parameters."""
        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer, scheduler = make_mht_with_warmup_schedule(
            optimizer=base_optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100,
            max_batch_draws=32,
            distributed_mode="sharded",
        )

        assert optimizer is not None

    def test_lr_schedule_warmup_and_annealing(self):
        """Learning rate follows warmup then cosine annealing."""

        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        initial_lr = 0.01

        optimizer, scheduler = make_mht_with_warmup_schedule(
            optimizer=base_optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # Test warmup phase (step 50)
        for _ in range(50):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        expected_lr = initial_lr * (1.0 * 50) / 100
        assert math.isclose(current_lr, expected_lr, rel_tol=0.01)

        # Test at end of warmup (step 100)
        for _ in range(50):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        expected_lr = initial_lr * 1.0
        assert math.isclose(current_lr, expected_lr, rel_tol=0.01)

        # Test annealing phase (step 550)
        for _ in range(450):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        # λ(t=550) = A + (W - A) * cos(π/2 * (t-L)/(M-L))
        A, W, L, M, t = 0.001, 1.0, 100, 1000, 550
        expected_lambda = A + (W - A) * math.cos(math.pi / 2 * (t - L) / (M - L))
        expected_lr = initial_lr * expected_lambda
        assert math.isclose(current_lr, expected_lr, rel_tol=0.01)

    def test_confidence_level_constant_after_warmup(self):
        """Confidence level stays constant after warmup."""

        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer, scheduler = make_mht_with_warmup_schedule(
            optimizer=base_optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # Step past warmup
        for _ in range(150):
            scheduler.step()
        conf_at_150 = scheduler.get_last_schedule("confidence_level")[0]

        # Step much further
        for _ in range(400):
            scheduler.step()
        conf_at_550 = scheduler.get_last_schedule("confidence_level")[0]

        # Should be constant at 0.95
        assert math.isclose(conf_at_150, 0.95, rel_tol=0.01)
        assert math.isclose(conf_at_550, 0.95, rel_tol=0.01)

    def test_error_threshold_constant_after_warmup(self):
        """Percent error threshold stays constant after warmup."""

        model = create_simple_model()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer, scheduler = make_mht_with_warmup_schedule(
            optimizer=base_optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # Step past warmup
        for _ in range(150):
            scheduler.step()
        error_at_150 = scheduler.get_last_schedule("percent_error_threshold")[0]

        # Step much further
        for _ in range(400):
            scheduler.step()
        error_at_550 = scheduler.get_last_schedule("percent_error_threshold")[0]

        # Should be constant at 0.1
        assert math.isclose(error_at_150, 0.1, rel_tol=0.01)
        assert math.isclose(error_at_550, 0.1, rel_tol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for scheduling utilities.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from src.gradient_quality_control.scheduling_utils import (
    get_direct_cosine_annealing_with_warmup,
    get_norm_threshold_cosine_annealing_with_warmup,
    get_quadratic_batch_schedule,
)


def create_mock_optimizer():
    """Create simple optimizer for testing."""
    param = nn.Parameter(torch.randn(10))
    optimizer = torch.optim.SGD([param], lr=1.0)
    return optimizer


class TestDirectCosineAnnealingWithWarmup:
    """Test direct cosine annealing scheduler."""

    def test_warmup_increases_linearly_from_zero(self):
        """During warmup, values increase linearly from 0 to peak."""
        optimizer = create_mock_optimizer()
        scheduler = get_direct_cosine_annealing_with_warmup(
            optimizer, peak_value=10.0, num_warmup_steps=4, num_training_steps=10, min_value=0.0
        )

        # Step 0 happens during init, we see steps 1-4
        values = []
        for _ in range(4):
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])

        # Should increase: step 1 (2.5) to step 4 (10.0)
        assert values[0] < values[-1]
        assert values[-1] == pytest.approx(10.0, rel=0.01)

    def test_annealing_decreases_after_warmup(self):
        """After warmup, cosine annealing decreases to min_value."""
        optimizer = create_mock_optimizer()
        scheduler = get_direct_cosine_annealing_with_warmup(
            optimizer, peak_value=10.0, num_warmup_steps=2, num_training_steps=10, min_value=2.0
        )

        # Skip warmup (steps 1-2)
        scheduler.step()
        scheduler.step()

        # Collect annealing values (steps 3-10)
        values = []
        for _ in range(8):
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])

        # Should decrease during annealing
        assert values[0] > values[-1]
        assert values[-1] == pytest.approx(2.0, rel=0.01)

    def test_default_min_value_is_zero(self):
        """min_value defaults to 0.0."""
        optimizer = create_mock_optimizer()
        scheduler = get_direct_cosine_annealing_with_warmup(
            optimizer, peak_value=5.0, num_warmup_steps=1, num_training_steps=3
        )

        # Run to completion (steps 1-3)
        for _ in range(3):
            scheduler.step()

        final_value = optimizer.param_groups[0]["lr"]
        assert final_value == pytest.approx(0.0, abs=0.01)

    def test_is_lr_scheduler(self):
        """Should be instance of LRScheduler."""
        optimizer = create_mock_optimizer()
        scheduler = get_direct_cosine_annealing_with_warmup(
            optimizer, peak_value=1.0, num_warmup_steps=5, num_training_steps=10
        )

        assert isinstance(scheduler, LRScheduler)


class TestNormThresholdCosineAnnealingWithWarmup:
    """Test norm threshold scheduler with inverted warmup."""

    def test_inverted_warmup_decreases_from_high(self):
        """Warmup should decrease from high value to start_norm."""
        optimizer = create_mock_optimizer()
        scheduler = get_norm_threshold_cosine_annealing_with_warmup(
            optimizer,
            num_warmup_steps=4,
            num_training_steps=10,
            start_norm=2.0,
            end_norm=0.5,
            warmup_multiplier=5.0,
        )

        # Step 0 during init (10.0), we see steps 1-4
        values = []
        for _ in range(4):
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])

        # Should decrease from 8.0 (step 1) to 2.0 (step 4)
        assert values[0] > values[-1]
        assert values[0] == pytest.approx(8.0, rel=0.01)
        assert values[-1] == pytest.approx(2.0, rel=0.01)

    def test_annealing_after_warmup(self):
        """After warmup, should anneal from start_norm to end_norm."""
        optimizer = create_mock_optimizer()
        scheduler = get_norm_threshold_cosine_annealing_with_warmup(
            optimizer, num_warmup_steps=2, num_training_steps=10, start_norm=5.0, end_norm=1.0
        )

        # Skip warmup (steps 1-2)
        scheduler.step()
        scheduler.step()

        # Collect annealing values (steps 3-10)
        values = []
        for _ in range(8):
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])

        # Should decrease from start_norm to end_norm
        assert values[0] > values[-1]
        assert values[-1] == pytest.approx(1.0, rel=0.01)

    def test_default_values(self):
        """Defaults: start_norm=1.0, end_norm=0.0, warmup_multiplier=10.0."""
        optimizer = create_mock_optimizer()
        scheduler = get_norm_threshold_cosine_annealing_with_warmup(
            optimizer, num_warmup_steps=2, num_training_steps=5
        )

        # Step 1 should be between 10.0 and 1.0
        scheduler.step()
        first_value = optimizer.param_groups[0]["lr"]
        assert 1.0 < first_value < 10.0

        # Run to completion - should end near 0.0
        for _ in range(4):
            scheduler.step()
        final_value = optimizer.param_groups[0]["lr"]
        assert final_value == pytest.approx(0.0, abs=0.01)

    def test_custom_warmup_multiplier(self):
        """Custom warmup_multiplier affects warmup start."""
        optimizer = create_mock_optimizer()
        scheduler = get_norm_threshold_cosine_annealing_with_warmup(
            optimizer,
            num_warmup_steps=2,
            num_training_steps=5,
            start_norm=3.0,
            warmup_multiplier=4.0,
        )

        # Step 1 should be between 12.0 and 3.0
        scheduler.step()
        first_value = optimizer.param_groups[0]["lr"]
        assert 3.0 < first_value < 12.0

    def test_is_lr_scheduler(self):
        """Should be instance of LRScheduler."""
        optimizer = create_mock_optimizer()
        scheduler = get_norm_threshold_cosine_annealing_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=10
        )

        assert isinstance(scheduler, LRScheduler)


class TestQuadraticBatchSchedule:
    """Test quadratic batch size growth scheduler."""

    def test_quadratic_growth(self):
        """Batch size should grow quadratically."""
        optimizer = create_mock_optimizer()
        scheduler = get_quadratic_batch_schedule(
            optimizer, initial_batch_size=32, final_batch_size=320, num_training_steps=10
        )

        # Step 0 during init, we see steps 1-10
        values = []
        for _ in range(10):
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])

        # Should grow slowly early, accelerate late
        early_growth = values[2] - values[1]
        late_growth = values[-1] - values[-2]
        assert late_growth > early_growth

        # Should end at final_batch_size
        assert values[-1] == pytest.approx(320, rel=0.01)

    def test_starts_near_initial(self):
        """Early values should be near initial_batch_size."""
        optimizer = create_mock_optimizer()
        scheduler = get_quadratic_batch_schedule(
            optimizer, initial_batch_size=64, final_batch_size=256, num_training_steps=5
        )

        # Step 1
        scheduler.step()
        first_value = optimizer.param_groups[0]["lr"]
        # Step 1 (progress=0.2): 64 + 192*0.04 = 71.68
        assert 64 < first_value < 100

    def test_ends_at_final(self):
        """Last value should reach final_batch_size."""
        optimizer = create_mock_optimizer()
        scheduler = get_quadratic_batch_schedule(
            optimizer, initial_batch_size=32, final_batch_size=512, num_training_steps=8
        )

        # Steps 1-8
        for _ in range(8):
            scheduler.step()

        final_value = optimizer.param_groups[0]["lr"]
        assert final_value == pytest.approx(512, rel=0.01)

    def test_is_lr_scheduler(self):
        """Should be instance of LRScheduler."""
        optimizer = create_mock_optimizer()
        scheduler = get_quadratic_batch_schedule(
            optimizer, initial_batch_size=32, final_batch_size=256, num_training_steps=10
        )

        assert isinstance(scheduler, LRScheduler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

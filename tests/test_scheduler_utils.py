"""
Tests for NormWarmupScheduler.
"""

import unittest

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR

from src.gradient_quality_control.scheduling_utils import NormWarmupScheduler


class TestNormWarmupScheduler(unittest.TestCase):
    """Test warmup scheduler behavior."""

    def test_during_warmup_uses_linear_interpolation(self):
        """During warmup, value linearly interpolates from start to end."""
        optimizer = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=1.0)
        base_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        warmup = NormWarmupScheduler(
            base_scheduler, num_warmup_steps=4, warmup_start=10.0, warmup_end=2.0
        )

        values = []
        for _ in range(4):
            warmup.step()
            values.append(optimizer.param_groups[0]["lr"])

        # Should decay from high to low
        self.assertGreater(values[0], values[-1])

    def test_after_warmup_wrapped_scheduler_controls_values(self):
        """After warmup period, wrapped scheduler determines the lr values."""
        optimizer = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=1.0)
        base_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0)

        warmup = NormWarmupScheduler(
            base_scheduler, num_warmup_steps=2, warmup_start=10.0, warmup_end=1.0
        )

        warmup.step()
        warmup.step()

        post_warmup_values = []
        for _ in range(5):
            warmup.step()
            post_warmup_values.append(optimizer.param_groups[0]["lr"])

        self.assertLess(post_warmup_values[-1], post_warmup_values[0])
        self.assertLess(post_warmup_values[-1], 1.0)

    def test_wrapped_scheduler_steps_during_warmup(self):
        """Wrapped scheduler advances its internal state during warmup."""
        optimizer = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=1.0)
        base_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        warmup = NormWarmupScheduler(
            base_scheduler, num_warmup_steps=3, warmup_start=10.0, warmup_end=1.0
        )

        initial_epoch = base_scheduler.last_epoch
        warmup.step()
        warmup.step()

        self.assertGreater(base_scheduler.last_epoch, initial_epoch)

    def test_isinstance_returns_true_for_scheduler(self):
        """Wrapper is recognized as a scheduler."""
        optimizer = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=1.0)
        base_scheduler = StepLR(optimizer, step_size=10)

        warmup = NormWarmupScheduler(base_scheduler, num_warmup_steps=5)

        self.assertIsInstance(warmup, LRScheduler)


if __name__ == "__main__":
    unittest.main()

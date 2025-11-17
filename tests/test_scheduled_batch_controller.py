"""
Tests for OptimizerWrapperSBC (Scheduled Batch Controller).
"""

import unittest
from unittest.mock import Mock

import torch
import torch.nn as nn

from src.gradient_quality_control.scheduled_batch_controller import OptimizerWrapperSBC


def create_mock_optimizer(num_params=3, param_shape=(10,)):
    """Create mock optimizer with real parameters."""
    params = [nn.Parameter(torch.randn(param_shape)) for _ in range(num_params)]
    mock_opt = Mock()
    mock_opt.param_groups = [{"params": params}]
    mock_opt.step = Mock(return_value=None)
    mock_opt.zero_grad = Mock()
    return mock_opt, params


class TestTargetDrawsCalculation(unittest.TestCase):
    """Test target_draws property behavior."""

    def test_target_draws_rounds_to_nearest(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)

        # 100 / 32 = 3.125 -> rounds to 3
        wrapper.param_groups[0]["lr"] = 100.0
        self.assertEqual(wrapper.target_draws, 3)

        # 120 / 32 = 3.75 -> rounds to 4
        wrapper.param_groups[0]["lr"] = 120.0
        self.assertEqual(wrapper.target_draws, 4)

    def test_target_draws_minimum_is_one(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=100)

        # 10 / 100 = 0.1 -> should be at least 1
        wrapper.param_groups[0]["lr"] = 10.0
        self.assertEqual(wrapper.target_draws, 1)

    def test_default_initial_batch_size(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=64)

        # Default: logical = physical, so target_draws = 1
        self.assertEqual(wrapper.target_draws, 1)

    def test_custom_initial_batch_size(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(
            mock_opt, physical_batch_size=32, initial_logical_batch_size=128
        )

        # 128 / 32 = 4
        self.assertEqual(wrapper.target_draws, 4)


class TestControllerBehavior(unittest.TestCase):
    """Test stepping behavior based on scheduled batch size."""

    def test_steps_immediately_when_target_is_one(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)
        # Default: logical = physical, target_draws = 1

        for p in params:
            p.grad = torch.randn_like(p)

        result = wrapper.step()

        self.assertTrue(result)
        mock_opt.step.assert_called_once()

    def test_accumulates_until_target_draws(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(
            mock_opt, physical_batch_size=32, initial_logical_batch_size=96  # 96/32 = 3 draws
        )

        # First draw
        for p in params:
            p.grad = torch.randn_like(p)
        result1 = wrapper.step()
        self.assertFalse(result1)

        # Second draw
        for p in params:
            p.grad = torch.randn_like(p)
        result2 = wrapper.step()
        self.assertFalse(result2)

        # Third draw - should step
        for p in params:
            p.grad = torch.randn_like(p)
        result3 = wrapper.step()
        self.assertTrue(result3)
        mock_opt.step.assert_called_once()

    def test_force_steps_at_max_draws(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(
            mock_opt,
            physical_batch_size=32,
            initial_logical_batch_size=1000,  # Would need 31 draws
            max_batch_draws=5,
        )

        # Accumulate up to max_draws
        for i in range(5):
            for p in params:
                p.grad = torch.randn_like(p)
            result = wrapper.step()
            if i < 4:
                self.assertFalse(result)

        # Fifth draw should force step
        self.assertTrue(result)
        mock_opt.step.assert_called_once()

    def test_responds_to_scheduler_changes(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)

        # Start with target_draws = 1
        for p in params:
            p.grad = torch.randn_like(p)
        wrapper.step()  # Steps immediately

        # Scheduler changes logical batch size
        wrapper.param_groups[0]["lr"] = 64.0  # Now target_draws = 2

        # First draw after change
        for p in params:
            p.grad = torch.randn_like(p)
        result1 = wrapper.step()
        self.assertFalse(result1)

        # Second draw - should step
        for p in params:
            p.grad = torch.randn_like(p)
        result2 = wrapper.step()
        self.assertTrue(result2)


class TestStatistics(unittest.TestCase):
    """Test statistics contract."""

    def test_statistics_contains_required_keys(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)
        stats = wrapper.statistics()

        self.assertIn("target_draws", stats)
        self.assertIn("target_logical_batch_size", stats)
        self.assertIn("physical_batch_size", stats)
        self.assertIn("batches", stats)
        self.assertIn("steps", stats)
        self.assertIn("num_draws", stats)

    def test_physical_batch_size_in_statistics(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=128)
        stats = wrapper.statistics()

        self.assertEqual(stats["physical_batch_size"], 128)

    def test_target_values_update_with_scheduler(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)

        wrapper.param_groups[0]["lr"] = 256.0  # 256/32 = 8 draws
        stats = wrapper.statistics()

        self.assertEqual(stats["target_logical_batch_size"], 256.0)
        self.assertEqual(stats["target_draws"], 8)


if __name__ == "__main__":
    unittest.main()

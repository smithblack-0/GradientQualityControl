"""
Tests for OptimizerWrapperSBC (Scheduled Batch Controller).
"""

from unittest.mock import Mock

import pytest
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


class TestTargetDrawsCalculation:
    """Test target_draws property behavior."""

    def test_target_draws_rounds_to_nearest(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)

        # 100 / 32 = 3.125 -> rounds to 3
        wrapper.param_groups[0]["lr"] = 100.0
        assert wrapper.target_draws == 3

        # 120 / 32 = 3.75 -> rounds to 4
        wrapper.param_groups[0]["lr"] = 120.0
        assert wrapper.target_draws == 4

    def test_target_draws_minimum_is_one(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=100)

        # 10 / 100 = 0.1 -> should be at least 1
        wrapper.param_groups[0]["lr"] = 10.0
        assert wrapper.target_draws == 1


class TestControllerBehavior:
    """Test stepping behavior based on scheduled batch size."""

    def test_steps_immediately_when_target_is_one(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)
        # Default: logical = physical, target_draws = 1

        for p in params:
            p.grad = torch.randn_like(p)

        result = wrapper.step()

        assert result is True
        mock_opt.step.assert_called_once()

    def test_accumulates_until_target_draws(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(
            mock_opt,
            physical_batch_size=32,
        )
        wrapper.param_groups[0]["lr"] = 96  # 96/32 = 3 draws

        # First draw
        for p in params:
            p.grad = torch.randn_like(p)
        result1 = wrapper.step()
        assert result1 is False

        # Second draw
        for p in params:
            p.grad = torch.randn_like(p)
        result2 = wrapper.step()
        assert result2 is False

        # Third draw - should step
        for p in params:
            p.grad = torch.randn_like(p)
        result3 = wrapper.step()
        assert result3 is True
        mock_opt.step.assert_called_once()

    def test_force_steps_at_max_draws(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(
            mock_opt,
            physical_batch_size=32,
            max_batch_draws=5,
        )
        wrapper.param_groups[0]["lr"] = 1000  # would need 31 draws

        # Accumulate up to max_draws
        for i in range(5):
            for p in params:
                p.grad = torch.randn_like(p)
            result = wrapper.step()
            if i < 4:
                assert result is False

        # Fifth draw should force step
        assert result is True
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
        assert result1 is False

        # Second draw - should step
        for p in params:
            p.grad = torch.randn_like(p)
        result2 = wrapper.step()
        assert result2 is True


class TestStatistics:
    """Test statistics contract."""

    def test_statistics_contains_required_keys(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)
        stats = wrapper.statistics()

        assert "target_draws" in stats
        assert "target_logical_batch_size" in stats
        assert "physical_batch_size" in stats
        assert "batches" in stats
        assert "steps" in stats
        assert "num_draws" in stats

    def test_physical_batch_size_in_statistics(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=128)
        stats = wrapper.statistics()

        assert stats["physical_batch_size"] == 128

    def test_target_values_update_with_scheduler(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperSBC(mock_opt, physical_batch_size=32)

        wrapper.param_groups[0]["lr"] = 256.0  # 256/32 = 8 draws
        stats = wrapper.statistics()

        assert stats["target_logical_batch_size"] == 256.0
        assert stats["target_draws"] == 8


if __name__ == "__main__":
    pytest.main([__file__])

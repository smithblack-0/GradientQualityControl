"""
Tests for OptimizerWrapperGNTS (Gradient Norm Threshold Scheduler).
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from src.gradient_quality_control.gradient_norm_threshold_scheduler import (
    OptimizerWrapperGNTS,
)


def create_mock_optimizer(num_params=3, param_shape=(10,)):
    """Create mock optimizer with real parameters."""
    params = [nn.Parameter(torch.randn(param_shape)) for _ in range(num_params)]
    mock_opt = Mock()
    mock_opt.param_groups = [{"params": params}]
    mock_opt.step = Mock(return_value=None)
    mock_opt.zero_grad = Mock()
    return mock_opt, params


class TestNormThresholdProperty:
    """Test norm_threshold property behavior."""

    def test_norm_threshold_reads_from_param_groups(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNTS(mock_opt)
        wrapper.param_groups[0]["lr"] = 5.0

        assert wrapper.norm_threshold == 5.0

    def test_norm_threshold_defaults_to_one(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNTS(mock_opt)

        assert wrapper.norm_threshold == 1.0


class TestControllerBehavior:
    """Test stepping behavior based on gradient norm threshold."""

    def test_steps_when_norm_below_threshold(self):
        mock_opt, params = create_mock_optimizer(num_params=1, param_shape=(4,))
        wrapper = OptimizerWrapperGNTS(mock_opt)
        wrapper.param_groups[0]["lr"] = 10.0  # High threshold

        # Small gradient norm
        params[0].grad = torch.tensor([1.0, 0.0, 0.0, 0.0])

        result = wrapper.step()

        assert result is True
        mock_opt.step.assert_called_once()

    def test_accumulates_when_norm_above_threshold(self):
        mock_opt, params = create_mock_optimizer(num_params=1, param_shape=(4,))
        wrapper = OptimizerWrapperGNTS(mock_opt)
        wrapper.param_groups[0]["lr"] = 0.1  # Low threshold

        # Large gradient norm
        params[0].grad = torch.tensor([10.0, 10.0, 10.0, 10.0])

        result = wrapper.step()

        assert result is False
        mock_opt.step.assert_not_called()

    def test_force_steps_at_max_draws(self):
        mock_opt, params = create_mock_optimizer(num_params=1, param_shape=(4,))
        wrapper = OptimizerWrapperGNTS(mock_opt, max_batch_draws=3)
        wrapper.param_groups[0]["lr"] = 0.001  # Very low threshold, never met

        # Large gradient that won't meet threshold
        params[0].grad = torch.tensor([100.0, 100.0, 100.0, 100.0])

        # Accumulate up to max_draws
        wrapper.step()  # 1st
        wrapper.step()  # 2nd
        result = wrapper.step()  # 3rd = max_draws, force step

        assert result is True
        mock_opt.step.assert_called_once()

    def test_gradient_accumulation_reduces_norm(self):
        mock_opt, params = create_mock_optimizer(num_params=1, param_shape=(2,))
        wrapper = OptimizerWrapperGNTS(mock_opt)
        wrapper.param_groups[0]["lr"] = 5.0  # Threshold

        # Initial gradient with high norm
        params[0].grad = torch.tensor([6.0, 8.0])  # norm = 10

        # First step: norm=10 > 5, accumulate
        result1 = wrapper.step()
        assert result1 is False

        # Add more gradients (they accumulate)
        # After averaging, norm should decrease
        params[0].grad = torch.tensor([9.0, 12.0])  # accumulated sum

        # Second step: norm=15/2=7.5 > 5, still accumulate
        result2 = wrapper.step()
        assert result2 is False

        # More accumulation
        params[0].grad = torch.tensor([12.0, 16.0])  # accumulated sum

        # Third step: norm=20/3=6.67 > 5, still accumulate
        result3 = wrapper.step()
        assert result3 is False

        # Keep going
        params[0].grad = torch.tensor([15.0, 20.0])  # accumulated sum

        # Fourth step: norm=25/4=6.25 > 5, still accumulate
        result4 = wrapper.step()
        assert result4 is False

        # Finally
        params[0].grad = torch.tensor([18.0, 24.0])  # accumulated sum

        # Fifth step: norm=30/5=6 > 5, still accumulate
        result5 = wrapper.step()
        assert result5 is False

        # Add smaller gradient contribution
        params[0].grad = torch.tensor([20.0, 26.0])  # accumulated sum

        # Sixth step: norm ~ 32.8/6 ~ 5.47 > 5, still not quite
        result6 = wrapper.step()
        assert result6 is False

        # Even smaller
        params[0].grad = torch.tensor([21.0, 28.0])  # accumulated sum

        # Seventh: norm ~ 35/7 = 5.0, exactly at threshold
        # Let's push below
        params[0].grad = torch.tensor([21.0, 27.0])  # norm ~ 34.2/7 ~ 4.9 < 5

        result7 = wrapper.step()
        assert result7 is True

    def test_works_with_real_scheduler(self):
        """Can attach actual PyTorch scheduler."""
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNTS(mock_opt)

        scheduler = torch.optim.lr_scheduler.StepLR(wrapper, step_size=1, gamma=0.5)

        initial = wrapper.norm_threshold
        scheduler.step()

        assert wrapper.norm_threshold == pytest.approx(initial * 0.5)


class TestStatistics:
    """Test statistics contract."""

    def test_statistics_contains_required_keys(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNTS(mock_opt)
        stats = wrapper.statistics()

        assert "norm_threshold" in stats
        assert "batches" in stats
        assert "steps" in stats
        assert "num_draws" in stats

    def test_norm_threshold_reflected_in_statistics(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNTS(mock_opt)
        wrapper.param_groups[0]["lr"] = 7.5
        stats = wrapper.statistics()

        assert stats["norm_threshold"] == 7.5


if __name__ == "__main__":
    pytest.main([__file__])

"""
Tests for OptimizerWrapperMHT (Metric Hypothesis Test controller).
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from src.gradient_quality_control.metric_hypothesis_test import OptimizerWrapperMHT


def create_mock_optimizer(num_params=3, param_shape=(10,)):
    """Create mock optimizer with real parameters."""
    params = [nn.Parameter(torch.randn(param_shape)) for _ in range(num_params)]
    mock_opt = Mock()
    mock_opt.param_groups = [{"params": params}]
    mock_opt.step = Mock(return_value=None)
    mock_opt.zero_grad = Mock()
    return mock_opt, params


class TestHelperFunctions:
    """Test static hypothesis test function atomically."""

    def test_rejects_with_insufficient_samples(self):
        """Need at least 2 samples for t-test."""
        metrics = [1.0]
        running_avg = 1.0
        result = OptimizerWrapperMHT._is_null_hypothesis_rejected(
            metrics, running_avg, confidence=0.95, error_tolerance=0.1
        )
        assert not result

    def test_rejects_null_with_tight_confidence_interval(self):
        """Identical metrics = zero variance = tight CI = reject null."""
        metrics = [1.0, 1.0, 1.0, 1.0, 1.0]
        running_avg = 1.0
        result = OptimizerWrapperMHT._is_null_hypothesis_rejected(
            metrics, running_avg, confidence=0.98, error_tolerance=0.03
        )
        assert result

    def test_accepts_null_with_wide_confidence_interval(self):
        """High variance = wide CI = accept null (don't step)."""
        metrics = [0.5, 1.5, 0.5, 1.5]  # High variance
        running_avg = 1.0
        result = OptimizerWrapperMHT._is_null_hypothesis_rejected(
            metrics, running_avg, confidence=0.98, error_tolerance=0.01  # Strict tolerance
        )
        assert not result

    def test_more_samples_tighten_confidence_interval(self):
        """More consistent samples = tighter CI."""
        running_avg = 1.0
        few_samples = [1.0, 1.0, 1.0]
        many_samples = [1.0] * 20

        result_few = OptimizerWrapperMHT._is_null_hypothesis_rejected(
            few_samples, running_avg, confidence=0.98, error_tolerance=0.03
        )
        result_many = OptimizerWrapperMHT._is_null_hypothesis_rejected(
            many_samples, running_avg, confidence=0.98, error_tolerance=0.03
        )

        # Both true since variance is 0
        assert result_few
        assert result_many

    def test_returns_false_for_zero_mean(self):
        """Zero mean edge case handled."""
        metrics = [0.0, 0.0]
        running_avg = 0.0
        result = OptimizerWrapperMHT._is_null_hypothesis_rejected(
            metrics, running_avg, confidence=0.98, error_tolerance=0.03
        )
        assert not result


class TestControllerBehavior:
    """Test stepping behavior based on hypothesis test."""

    def test_accumulates_with_single_sample(self):
        """Can't compute t-test with one sample."""
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperMHT(mock_opt)

        for p in params:
            p.grad = torch.randn_like(p)

        result = wrapper.step(metric=1.0)

        assert not result
        mock_opt.step.assert_not_called()

    def test_steps_with_consistent_metrics(self):
        """Consistent metrics = tight CI = step."""
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperMHT(mock_opt, confidence=0.98, error_tolerance=0.03)

        # Feed consistent metrics
        stepped = False
        for _ in range(10):
            for p in params:
                p.grad = torch.randn_like(p)
            if wrapper.step(metric=1.0):
                stepped = True
                break

        assert stepped

    def test_accumulates_with_variable_metrics(self):
        """Variable metrics = wide CI = accumulate."""
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperMHT(
            mock_opt, confidence=0.98, error_tolerance=0.001, max_batch_draws=100
        )

        # Feed highly variable metrics
        for p in params:
            p.grad = torch.randn_like(p)
        result1 = wrapper.step(metric=0.1)

        for p in params:
            p.grad = torch.randn_like(p)
        result2 = wrapper.step(metric=10.0)

        for p in params:
            p.grad = torch.randn_like(p)
        result3 = wrapper.step(metric=0.1)

        # Should still be accumulating due to high variance
        assert not result1
        assert not result2
        assert not result3

    def test_force_steps_at_max_draws(self):
        """Steps when max_draws reached."""
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperMHT(
            mock_opt, confidence=0.999, error_tolerance=0.0001, max_batch_draws=3
        )

        # Very strict params that won't be met
        for p in params:
            p.grad = torch.randn_like(p)
        wrapper.step(metric=1.0)  # 1st

        for p in params:
            p.grad = torch.randn_like(p)
        wrapper.step(metric=2.0)  # 2nd

        for p in params:
            p.grad = torch.randn_like(p)
        result = wrapper.step(metric=3.0)  # 3rd = max_draws

        assert result
        mock_opt.step.assert_called_once()

    def test_ema_updates_after_step(self):
        """Running average updates when optimizer steps."""
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperMHT(mock_opt, confidence=0.98, error_tolerance=0.5, ema_alpha=0.1)

        # Feed metrics until step
        for _ in range(20):
            for p in params:
                p.grad = torch.randn_like(p)
            if wrapper.step(metric=5.0):
                # After stepping, EMA should be updated
                assert wrapper.running_avg_metric is not None
                break

        # EMA should reflect the fact we’ve had at least one metric update
        assert wrapper.running_avg_metric is not None


class TestStatistics:
    """Test statistics contract."""

    def test_statistics_contains_required_keys(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperMHT(mock_opt)
        stats = wrapper.statistics()

        assert "running_avg_metric" in stats
        assert "batches" in stats
        assert "steps" in stats
        assert "num_draws" in stats

    def test_running_avg_initially_none(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperMHT(mock_opt)
        stats = wrapper.statistics()

        # Before any metrics, should be None
        assert stats["running_avg_metric"] is None


if __name__ == "__main__":
    pytest.main([__file__])

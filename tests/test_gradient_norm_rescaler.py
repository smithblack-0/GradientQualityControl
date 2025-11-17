"""
Tests for OptimizerWrapperGNR (Gradient Norm Rescaler).
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from src.gradient_quality_control.gradient_norm_rescalar import OptimizerWrapperGNR


def create_mock_optimizer(num_params=3, param_shape=(10,)):
    """Create mock optimizer with real parameters."""
    params = [nn.Parameter(torch.randn(param_shape)) for _ in range(num_params)]
    mock_opt = Mock()
    mock_opt.param_groups = [{"params": params}]
    mock_opt.step = Mock(return_value=None)
    mock_opt.zero_grad = Mock()
    return mock_opt, params


class TestTargetNormProperty:
    """Test target_norm property behavior."""

    def test_target_norm_reads_from_param_groups(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNR(mock_opt)
        wrapper.param_groups[0]["lr"] = 2.5

        assert wrapper.target_norm == 2.5

    def test_target_norm_defaults_to_one(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNR(mock_opt)

        assert wrapper.target_norm == 1.0


class TestGlobalModeScaling:
    """Test gradient scaling in global mode."""

    def test_scales_gradients_to_target_norm(self):
        mock_opt, params = create_mock_optimizer(num_params=2, param_shape=(3,))
        wrapper = OptimizerWrapperGNR(mock_opt, mode="global")
        wrapper.param_groups[0]["lr"] = 10.0  # Target norm

        # Set gradients with known total norm = 5
        params[0].grad = torch.tensor([3.0, 0.0, 0.0])
        params[1].grad = torch.tensor([4.0, 0.0, 0.0])

        wrapper.step()

        # Total norm should be 10.0
        total_norm = torch.nn.utils.get_total_norm([p.grad for p in params])
        assert total_norm.item() == pytest.approx(10.0, rel=1e-4, abs=1e-4)

    def test_all_gradients_scaled_uniformly(self):
        mock_opt, params = create_mock_optimizer(num_params=3, param_shape=(2,))
        wrapper = OptimizerWrapperGNR(mock_opt, mode="global")
        wrapper.param_groups[0]["lr"] = 1.0

        params[0].grad = torch.tensor([1.0, 0.0])
        params[1].grad = torch.tensor([2.0, 0.0])
        params[2].grad = torch.tensor([3.0, 0.0])

        # Total norm = sqrt(1 + 4 + 9) = sqrt(14)
        original_ratios = [1.0 / 2.0, 2.0 / 3.0]

        wrapper.step()

        # Ratios between gradients should be preserved
        new_ratios = [
            params[0].grad[0].item() / params[1].grad[0].item(),
            params[1].grad[0].item() / params[2].grad[0].item(),
        ]
        assert new_ratios[0] == pytest.approx(original_ratios[0], rel=1e-5, abs=1e-5)
        assert new_ratios[1] == pytest.approx(original_ratios[1], rel=1e-5, abs=1e-5)


class TestIndependentModeScaling:
    """Test gradient scaling in independent mode."""

    def test_each_parameter_scaled_independently(self):
        mock_opt, params = create_mock_optimizer(num_params=2, param_shape=(3,))
        wrapper = OptimizerWrapperGNR(mock_opt, mode="independent")
        wrapper.param_groups[0]["lr"] = 1.0  # Target norm

        # Different norms for each parameter
        params[0].grad = torch.tensor([3.0, 4.0, 0.0])  # norm = 5
        params[1].grad = torch.tensor([0.0, 0.0, 10.0])  # norm = 10

        wrapper.step()

        # Each should have norm = 1.0
        norm_0 = params[0].grad.norm().item()
        norm_1 = params[1].grad.norm().item()

        assert norm_0 == pytest.approx(1.0, rel=1e-5, abs=1e-5)
        assert norm_1 == pytest.approx(1.0, rel=1e-5, abs=1e-5)

    def test_direction_preserved_after_scaling(self):
        mock_opt, params = create_mock_optimizer(num_params=1, param_shape=(3,))
        wrapper = OptimizerWrapperGNR(mock_opt, mode="independent")
        wrapper.param_groups[0]["lr"] = 2.0

        original = torch.tensor([3.0, 4.0, 0.0])
        params[0].grad = original.clone()

        wrapper.step()

        # Direction should be same (normalized vectors equal)
        original_dir = original / original.norm()
        new_dir = params[0].grad / params[0].grad.norm()

        torch.testing.assert_close(original_dir, new_dir)


class TestControllerBehavior:
    """Test general controller contract."""

    def test_always_returns_true(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperGNR(mock_opt)

        for p in params:
            p.grad = torch.randn_like(p)

        result = wrapper.step()
        assert result is True

    def test_always_steps_optimizer(self):
        mock_opt, params = create_mock_optimizer()
        wrapper = OptimizerWrapperGNR(mock_opt)

        for p in params:
            p.grad = torch.randn_like(p)

        wrapper.step()
        wrapper.step()
        wrapper.step()

        assert mock_opt.step.call_count == 3

    def test_skips_none_gradients(self):
        mock_opt, params = create_mock_optimizer(num_params=2)
        wrapper = OptimizerWrapperGNR(mock_opt, mode="independent")
        wrapper.param_groups[0]["lr"] = 1.0

        params[0].grad = torch.tensor([3.0, 4.0] + [0.0] * 8)  # norm = 5, shape (10,)
        params[1].grad = None

        # Should not raise
        wrapper.step()

        # First param scaled, second unchanged
        assert params[0].grad.norm().item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
        assert params[1].grad is None

    def test_rejects_invalid_mode(self):
        mock_opt, _ = create_mock_optimizer()

        with pytest.raises(ValueError):
            OptimizerWrapperGNR(mock_opt, mode="invalid")


class TestStatistics:
    """Test statistics contract."""

    def test_statistics_contains_required_keys(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNR(mock_opt)
        stats = wrapper.statistics()

        assert "target_norm" in stats
        assert "mode" in stats
        assert "batches" in stats
        assert "steps" in stats

    def test_mode_reflected_in_statistics(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNR(mock_opt, mode="independent")
        stats = wrapper.statistics()

        assert stats["mode"] == "independent"

    def test_target_norm_reflected_in_statistics(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNR(mock_opt)
        wrapper.param_groups[0]["lr"] = 3.14
        stats = wrapper.statistics()

        assert stats["target_norm"] == 3.14


if __name__ == "__main__":
    pytest.main([__file__])

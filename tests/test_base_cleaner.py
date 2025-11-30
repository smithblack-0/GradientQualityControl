"""
Tests for AbstractOptimizerWrapper base class.
Focus on external behavioral contracts.
"""

import json
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from src.gradient_quality_control.base import AbstractOptimizerWrapper


class ConcreteWrapper(AbstractOptimizerWrapper):
    """Minimal concrete implementation for testing base class."""

    def step(self):
        pass

    def zero_grad(self):
        pass


def create_mock_optimizer(num_params=3, param_shape=(10,)):
    """Create mock optimizer with real parameters for testing."""
    params = [nn.Parameter(torch.randn(param_shape)) for _ in range(num_params)]
    mock_opt = Mock()
    mock_opt.param_groups = [{"params": params}]
    mock_opt.step = Mock(return_value=None)
    mock_opt.zero_grad = Mock()
    return mock_opt, params


# ---------------------------------------------------------------------------
# Passthrough Tests
# ---------------------------------------------------------------------------


def test_optimizer_step_called():
    mock_opt, _ = create_mock_optimizer()
    wrapper = ConcreteWrapper(mock_opt)
    wrapper._take_optimizer_step()
    mock_opt.step.assert_called_once()


def test_optimizer_zero_grad_called():
    mock_opt, _ = create_mock_optimizer()
    wrapper = ConcreteWrapper(mock_opt)
    wrapper._take_optimizer_step()
    mock_opt.zero_grad.assert_called_once()


def test_closure_passed_to_optimizer():
    mock_opt, _ = create_mock_optimizer()
    wrapper = ConcreteWrapper(mock_opt)
    closure = Mock(return_value=1.5)
    wrapper._take_optimizer_step(closure)
    mock_opt.step.assert_called_once_with(closure)


def test_optimizer_return_value_cached():
    mock_opt, _ = create_mock_optimizer()
    mock_opt.step.return_value = 2.5
    wrapper = ConcreteWrapper(mock_opt)
    wrapper._take_optimizer_step()
    assert wrapper.last_optimizer_result == 2.5


def test_unknown_attributes_routed_to_optimizer():
    mock_opt, _ = create_mock_optimizer()
    mock_opt.custom_attr = "test_value"
    wrapper = ConcreteWrapper(mock_opt)
    assert wrapper.custom_attr == "test_value"


# ---------------------------------------------------------------------------
# Gradient Normalization
# ---------------------------------------------------------------------------


def test_gradients_averaged_by_draw_count():
    """After accumulating draws, gradients divided by num_draws."""
    mock_opt, params = create_mock_optimizer(num_params=1, param_shape=(4,))
    wrapper = ConcreteWrapper(mock_opt)

    # Starts with 1 draw. Add 3 more = 4 total
    wrapper._take_batch_step()
    wrapper._take_batch_step()
    wrapper._take_batch_step()

    params[0].grad = torch.tensor([4.0, 8.0, 12.0, 16.0])
    wrapper._take_optimizer_step()

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(params[0].grad, expected)


def test_none_gradients_skipped():
    mock_opt, params = create_mock_optimizer(num_params=2, param_shape=(3,))
    wrapper = ConcreteWrapper(mock_opt)

    params[0].grad = torch.tensor([3.0, 6.0, 9.0])
    params[1].grad = None

    wrapper._take_batch_step()  # Now 2 draws
    wrapper._take_batch_step()  # Now 3 draws
    wrapper._take_optimizer_step()

    expected = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(params[0].grad, expected)
    assert params[1].grad is None


# ---------------------------------------------------------------------------
# Statistics Contract
# ---------------------------------------------------------------------------


def test_statistics_has_required_keys():
    mock_opt, _ = create_mock_optimizer()
    wrapper = ConcreteWrapper(mock_opt)
    stats = wrapper._get_base_statistics()

    assert "batches" in stats
    assert "steps" in stats
    assert "num_draws" in stats


def test_batch_step_increments_batches():
    mock_opt, _ = create_mock_optimizer()
    wrapper = ConcreteWrapper(mock_opt)
    initial = wrapper._get_base_statistics()["batches"]
    wrapper._take_batch_step()
    after = wrapper._get_base_statistics()["batches"]
    assert after == initial + 1


def test_optimizer_step_increments_steps():
    mock_opt, _ = create_mock_optimizer()
    wrapper = ConcreteWrapper(mock_opt)
    initial = wrapper._get_base_statistics()["steps"]
    wrapper._take_optimizer_step()
    after = wrapper._get_base_statistics()["steps"]
    assert after == initial + 1


# ---------------------------------------------------------------------------
# State Dict Contract Tests
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_behavior():
    """After save/load, wrapper behaves identically."""
    mock_opt1, params1 = create_mock_optimizer(num_params=1, param_shape=(3,))
    wrapper1 = ConcreteWrapper(mock_opt1)

    # Execute some operations
    wrapper1._take_batch_step()
    wrapper1._take_batch_step()
    params1[0].grad = torch.tensor([3.0, 4.0, 0.0])
    wrapper1._take_optimizer_step()
    wrapper1._take_batch_step()

    # Get statistics before save
    stats_before = wrapper1._get_base_statistics()

    # Save and load into new wrapper
    state = wrapper1.state_dict()
    mock_opt2, params2 = create_mock_optimizer(num_params=1, param_shape=(3,))
    wrapper2 = ConcreteWrapper(mock_opt2)
    wrapper2.load_state_dict(state)

    # Statistics should match
    stats_after = wrapper2._get_base_statistics()

    assert stats_before == stats_after


def test_continued_training_after_load():
    """Can continue training seamlessly after load."""
    mock_opt1, params1 = create_mock_optimizer(num_params=1, param_shape=(2,))
    wrapper1 = ConcreteWrapper(mock_opt1)

    wrapper1._take_batch_step()
    params1[0].grad = torch.tensor([2.0, 2.0])
    wrapper1._take_optimizer_step()

    state = wrapper1.state_dict()

    # Load and continue
    mock_opt2, params2 = create_mock_optimizer(num_params=1, param_shape=(2,))
    wrapper2 = ConcreteWrapper(mock_opt2)
    wrapper2.load_state_dict(state)

    # Should be able to continue taking steps
    wrapper2._take_batch_step()
    params2[0].grad = torch.tensor([4.0, 4.0])
    wrapper2._take_optimizer_step()  # Should not raise

    stats = wrapper2._get_base_statistics()
    assert stats["steps"] == 2  # One from before, one after


def test_state_dict_is_serializable():
    """State dict can be serialized (for checkpointing)."""
    mock_opt, _ = create_mock_optimizer()
    wrapper = ConcreteWrapper(mock_opt)

    state = wrapper.state_dict()

    # Should be JSON serializable (or at least dict of primitives)
    # This is a proxy for "can be saved to disk"
    try:
        # Convert tensors to floats if present
        serializable_state = {
            k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v
            for k, v in state.items()
            if k != "optimizer"
        }
        json.dumps(serializable_state)
    except (TypeError, ValueError):
        pytest.fail("State dict not serializable")


if __name__ == "__main__":
    pytest.main([__file__])

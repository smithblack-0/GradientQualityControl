"""
Tests for OptimizerWrapperGNS (Gradient Noise Scale controller).
"""
import unittest
from unittest.mock import Mock
import torch
import torch.nn as nn
import numpy as np
from src.gradient_quality_control.gradient_noise_scale import OptimizerWrapperGNS


def create_mock_optimizer(num_params=3, param_shape=(10,)):
    """Create mock optimizer with real parameters."""
    params = [nn.Parameter(torch.randn(param_shape)) for _ in range(num_params)]
    mock_opt = Mock()
    mock_opt.param_groups = [{'params': params}]
    mock_opt.step = Mock(return_value=None)
    mock_opt.zero_grad = Mock()
    return mock_opt, params


def create_simple_model():
    """Create a simple model for testing actual backward passes."""
    model = nn.Linear(10, 2)
    return model


class TestHelperFunctions(unittest.TestCase):
    """Test static/pure helper functions atomically."""

    def test_compute_gns_identical_norms(self):
        """Identical norms = zero variance = GNS of 0."""
        norms = [1.0, 1.0, 1.0, 1.0]
        result = OptimizerWrapperGNS.compute_gns_estimate(norms)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_compute_gns_known_values(self):
        """Verify GNS = var(norms) / mean(norms^2)."""
        norms = [1.0, 2.0, 3.0]
        # var = 2/3, mean_squared = 14/3, GNS = 2/14
        expected = 2.0 / 14.0
        result = OptimizerWrapperGNS.compute_gns_estimate(norms)
        self.assertAlmostEqual(result, expected, places=6)

    def test_compute_gns_high_variance(self):
        """Higher variance = higher GNS."""
        low_var = [10.0, 10.0, 10.0]
        high_var = [1.0, 10.0, 19.0]

        low_gns = OptimizerWrapperGNS.compute_gns_estimate(low_var)
        high_gns = OptimizerWrapperGNS.compute_gns_estimate(high_var)

        self.assertGreater(high_gns, low_gns)

    def test_attach_grad_norm_hook_adds_hook(self):
        """Hook is attached to parameter."""
        param = nn.Parameter(torch.randn(5))
        self.assertFalse(hasattr(param, '_has_grad_norm_hook'))

        OptimizerWrapperGNS._attach_grad_norm_hook(param)

        self.assertTrue(hasattr(param, '_has_grad_norm_hook'))
        self.assertTrue(param._has_grad_norm_hook)

    def test_attach_grad_norm_hook_idempotent(self):
        """Attaching hook twice doesn't add multiple hooks."""
        param = nn.Parameter(torch.randn(5))

        OptimizerWrapperGNS._attach_grad_norm_hook(param)
        OptimizerWrapperGNS._attach_grad_norm_hook(param)

        # Should still only have one hook
        self.assertTrue(param._has_grad_norm_hook)

    def test_get_independent_grad_norms(self):
        """Computes total norm from last_gradient_norm attributes."""
        params = [nn.Parameter(torch.randn(5)) for _ in range(3)]
        params[0].last_gradient_norm = torch.tensor(3.0)
        params[1].last_gradient_norm = torch.tensor(4.0)
        params[2].last_gradient_norm = torch.tensor(0.0)

        # Total norm = sqrt(3^2 + 4^2 + 0^2) = 5.0
        result = OptimizerWrapperGNS._get_independent_grad_norms(params)
        self.assertAlmostEqual(result, 5.0, places=5)

    def test_get_independent_grad_norms_raises_without_attribute(self):
        """Raises error if parameter missing last_gradient_norm."""
        params = [nn.Parameter(torch.randn(5))]

        with self.assertRaises(RuntimeError):
            OptimizerWrapperGNS._get_independent_grad_norms(params)


class TestControllerBehavior(unittest.TestCase):
    """Test controller stepping behavior with real backward passes."""

    def test_does_not_step_with_one_sample(self):
        """Need at least 2 samples for GNS computation."""
        model = create_simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        wrapper = OptimizerWrapperGNS(optimizer, noise_multiplier=1.0)

        # One backward pass
        x = torch.randn(1, 10)
        loss = model(x).sum()
        loss.backward()

        result = wrapper.step()

        self.assertFalse(result)

    def test_steps_with_consistent_gradients(self):
        """Consistent gradients (low GNS) should trigger step."""
        model = create_simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        wrapper = OptimizerWrapperGNS(optimizer, noise_multiplier=10.0)

        x = torch.randn(1, 10)
        stepped_at = None

        for i in range(5):
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            if wrapper.step():
                stepped_at = i
                break

        # Should step after 2nd sample (GNS computable, criterion met)
        self.assertIsNotNone(stepped_at)
        self.assertGreaterEqual(stepped_at, 1)  # Need at least 2 samples

    def test_force_step_at_max_draws(self):
        """Steps when max_draws reached."""
        model = create_simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        wrapper = OptimizerWrapperGNS(optimizer, noise_multiplier=0.0, max_batch_draws=3)

        # noise_multiplier=0 means criterion never met, but max_draws forces step
        for i in range(3):
            optimizer.zero_grad()
            x = torch.randn(1, 10)  # Different input each time
            loss = model(x).sum()
            loss.backward()
            result = wrapper.step()

        self.assertTrue(result)

    def test_returns_false_while_accumulating(self):
        """Returns False when still accumulating."""
        model = create_simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        wrapper = OptimizerWrapperGNS(optimizer, noise_multiplier=0.0, max_batch_draws=10)

        optimizer.zero_grad()
        x = torch.randn(1, 10)
        loss = model(x).sum()
        loss.backward()
        result = wrapper.step()

        self.assertFalse(result)


class TestStatistics(unittest.TestCase):
    """Test statistics contract."""

    def test_statistics_contains_required_keys(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNS(mock_opt, noise_multiplier=0.5)
        stats = wrapper.statistics()

        self.assertIn('noise_multiplier', stats)
        self.assertIn('estimated_gns', stats)
        self.assertIn('batches', stats)
        self.assertIn('steps', stats)
        self.assertIn('num_draws', stats)

    def test_noise_multiplier_reflected_in_statistics(self):
        mock_opt, _ = create_mock_optimizer()
        wrapper = OptimizerWrapperGNS(mock_opt, noise_multiplier=0.75)
        stats = wrapper.statistics()

        self.assertEqual(stats['noise_multiplier'], 0.75)


if __name__ == '__main__':
    unittest.main()
import pytest
import torch
from torch.optim import SGD, Adam
from numbers import Number
import math

from src.gradient_quality_control.optim_utils import (
    optimizer_extend,
    optimizer_get_collection,
    optimizer_multiply_gradients,
    optimizer_get_grad_norm,
    optimizer_get_raw_grad_norms
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def single_group_optimizer():
    """Simple optimizer with one param group"""
    param = torch.randn(10, 5, requires_grad=True)
    optimizer = SGD([param], lr=0.01)
    return optimizer, [param]


@pytest.fixture
def multi_group_optimizer():
    """Optimizer with multiple param groups"""
    param1 = torch.randn(10, 5, requires_grad=True)
    param2 = torch.randn(3, 7, requires_grad=True)
    param3 = torch.randn(2, requires_grad=True)

    optimizer = SGD([
        {'params': [param1], 'lr': 0.01},
        {'params': [param2], 'lr': 0.001},
        {'params': [param3], 'lr': 0.0001}
    ])
    return optimizer, [param1, param2, param3]


@pytest.fixture
def empty_group_optimizer():
    """Optimizer with empty param groups"""
    optimizer = SGD([{'params': [], 'lr': 0.01}])
    return optimizer


# ============================================================================
# Tests for optimizer_extend
# ============================================================================

class TestOptimizerExtend:

    def test_basic_extension(self, single_group_optimizer):
        """Test adding a new parameter to param groups"""
        optimizer, _ = single_group_optimizer
        result = optimizer_extend(optimizer, 'custom_param', 0.5)

        assert result is optimizer  # Returns same optimizer
        assert optimizer.param_groups[0]['custom_param'] == 0.5

    def test_extension_multiple_groups(self, multi_group_optimizer):
        """Test extending all param groups"""
        optimizer, _ = multi_group_optimizer
        optimizer_extend(optimizer, 'blubber', 0.01)

        for group in optimizer.param_groups:
            assert group['blubber'] == 0.01

    def test_existing_key_unchanged(self, single_group_optimizer):
        """Test that existing keys are not overwritten"""
        optimizer, _ = single_group_optimizer
        optimizer.param_groups[0]['custom_param'] = 0.9

        optimizer_extend(optimizer, 'custom_param', 0.5)
        assert optimizer.param_groups[0]['custom_param'] == 0.9

    def test_non_number_raises_typeerror(self, single_group_optimizer):
        """Test that non-numeric default values raise TypeError"""
        optimizer, _ = single_group_optimizer

        with pytest.raises(TypeError, match="Only numbers can be inserted"):
            optimizer_extend(optimizer, 'bad_param', "not a number")

    def test_empty_param_groups(self, empty_group_optimizer):
        """Test extending empty param groups"""
        optimizer = empty_group_optimizer
        optimizer_extend(optimizer, 'custom_param', 0.5)

        assert optimizer.param_groups[0]['custom_param'] == 0.5


# ============================================================================
# Tests for optimizer_get_collection
# ============================================================================

class TestOptimizerGetCollection:

    def test_get_learning_rates(self, multi_group_optimizer):
        """Test retrieving learning rates"""
        optimizer, _ = multi_group_optimizer
        lrs = optimizer_get_collection(optimizer, 'lr')

        assert lrs == [0.01, 0.001, 0.0001]

    def test_get_params(self, single_group_optimizer):
        """Test retrieving params"""
        optimizer, params = single_group_optimizer
        retrieved_params = optimizer_get_collection(optimizer, 'params')

        assert len(retrieved_params) == 1
        assert retrieved_params[0] == params

    def test_get_extended_parameter(self, multi_group_optimizer):
        """Test retrieving custom extended parameters"""
        optimizer, _ = multi_group_optimizer
        optimizer_extend(optimizer, 'blubber', 0.9)

        momentums = optimizer_get_collection(optimizer, 'blubber')
        assert momentums == [0.9, 0.9, 0.9]

    def test_nonexistent_key_raises_keyerror(self, single_group_optimizer):
        """Test that accessing non-existent key raises KeyError"""
        optimizer, _ = single_group_optimizer

        with pytest.raises(KeyError):
            optimizer_get_collection(optimizer, 'nonexistent_key')


# ============================================================================
# Tests for optimizer_multiply_gradients
# ============================================================================

class TestOptimizerMultiplyGradients:

    def test_basic_multiplication(self, single_group_optimizer):
        """Test multiplying gradients by a scalar"""
        optimizer, params = single_group_optimizer
        param = params[0]
        param.grad = torch.ones_like(param) * 2.0

        optimizer_multiply_gradients(optimizer, 3.0)

        assert torch.allclose(param.grad, torch.ones_like(param) * 6.0)

    def test_multiple_groups(self, multi_group_optimizer):
        """Test multiplying gradients across multiple param groups"""
        optimizer, params = multi_group_optimizer

        # Set gradients
        for param in params:
            param.grad = torch.ones_like(param) * 2.0

        optimizer_multiply_gradients(optimizer, 0.5)

        for param in params:
            assert torch.allclose(param.grad, torch.ones_like(param) * 1.0)

    def test_zero_multiplication(self, single_group_optimizer):
        """Test multiplying by zero"""
        optimizer, params = single_group_optimizer
        param = params[0]
        param.grad = torch.randn_like(param)

        optimizer_multiply_gradients(optimizer, 0.0)

        assert torch.allclose(param.grad, torch.zeros_like(param))

    def test_mixed_none_and_real_gradients(self, multi_group_optimizer):
        """Test that None gradients are skipped during multiplication"""
        optimizer, params = multi_group_optimizer

        # Set some gradients, leave others as None
        params[0].grad = torch.ones_like(params[0]) * 2.0
        # params[1].grad remains None
        params[2].grad = torch.ones_like(params[2]) * 3.0

        optimizer_multiply_gradients(optimizer, 2.0)

        # Check that real gradients were multiplied
        assert torch.allclose(params[0].grad, torch.ones_like(params[0]) * 4.0)
        assert torch.allclose(params[2].grad, torch.ones_like(params[2]) * 6.0)
        # Check that None gradient remains None
        assert params[1].grad is None

    def test_all_none_gradients(self, multi_group_optimizer):
        """Test that all None gradients doesn't error"""
        optimizer, params = multi_group_optimizer
        # All gradients are None by default

        # Should not raise
        optimizer_multiply_gradients(optimizer, 2.0)

        # All should still be None
        for param in params:
            assert param.grad is None


# ============================================================================
# Tests for optimizer_get_grad_norm
# ============================================================================

class TestOptimizerGetGradNorm:

    def test_single_group_known_norm(self, single_group_optimizer):
        """Test gradient norm with known values"""
        optimizer, params = single_group_optimizer
        param = params[0]

        # Set gradient to have known norm
        param.grad = torch.ones_like(param) * 2.0
        expected_norm = math.sqrt((param.grad ** 2).sum().item())

        result = optimizer_get_grad_norm(optimizer)
        assert result == pytest.approx(expected_norm, rel=1e-5)

    def test_multiple_groups_l2_combination(self, multi_group_optimizer):
        """Test that multiple groups are combined via L2 norm"""
        optimizer, params = multi_group_optimizer

        # Set gradients with known norms
        params[0].grad = torch.ones_like(params[0]) * 3.0  # norm = 3 * sqrt(50)
        params[1].grad = torch.ones_like(params[1]) * 4.0  # norm = 4 * sqrt(21)
        params[2].grad = torch.ones_like(params[2]) * 5.0  # norm = 5 * sqrt(2)

        # Compute individual group norms manually
        norm1 = math.sqrt((params[0].grad ** 2).sum().item())
        norm2 = math.sqrt((params[1].grad ** 2).sum().item())
        norm3 = math.sqrt((params[2].grad ** 2).sum().item())

        # Expected: L2 of group norms
        expected = math.sqrt(norm1 ** 2 + norm2 ** 2 + norm3 ** 2)

        result = optimizer_get_grad_norm(optimizer)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_zero_gradients(self, single_group_optimizer):
        """Test with zero gradients"""
        optimizer, params = single_group_optimizer
        params[0].grad = torch.zeros_like(params[0])

        result = optimizer_get_grad_norm(optimizer)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_all_none_gradients(self, multi_group_optimizer):
        """Test that all None gradients returns 0.0"""
        optimizer, params = multi_group_optimizer
        # All gradients are None by default

        result = optimizer_get_grad_norm(optimizer)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_mixed_none_and_real_gradients(self, multi_group_optimizer):
        """Test that None gradients are skipped in norm calculation"""
        optimizer, params = multi_group_optimizer

        # Set some gradients, leave others as None
        params[0].grad = torch.ones_like(params[0]) * 3.0
        # params[1].grad remains None
        params[2].grad = torch.ones_like(params[2]) * 4.0

        # Compute expected norms for non-None groups
        norm1 = math.sqrt((params[0].grad ** 2).sum().item())
        norm2 = 0.0  # Group with None gradients
        norm3 = math.sqrt((params[2].grad ** 2).sum().item())

        expected = math.sqrt(norm1 ** 2 + norm2 ** 2 + norm3 ** 2)

        result = optimizer_get_grad_norm(optimizer)
        assert result == pytest.approx(expected, rel=1e-5)



# ============================================================================
# Tests for optimizer_get_raw_grad_norms
# ============================================================================

class TestOptimizerGetRawGradNorms:

    def test_single_group(self, single_group_optimizer):
        """Test getting raw norms for single group"""
        optimizer, params = single_group_optimizer
        param = params[0]
        param.grad = torch.ones_like(param) * 2.0

        expected_norm = math.sqrt((param.grad ** 2).sum().item())
        result = optimizer_get_raw_grad_norms(optimizer)

        assert len(result) == 1
        assert result[0] == pytest.approx(expected_norm, rel=1e-5)

    def test_multiple_groups_returns_list(self, multi_group_optimizer):
        """Test that multiple groups return individual norms as list"""
        optimizer, params = multi_group_optimizer

        # Set gradients
        params[0].grad = torch.ones_like(params[0]) * 3.0
        params[1].grad = torch.ones_like(params[1]) * 4.0
        params[2].grad = torch.ones_like(params[2]) * 5.0

        # Compute expected norms
        expected_norms = [
            math.sqrt((params[0].grad ** 2).sum().item()),
            math.sqrt((params[1].grad ** 2).sum().item()),
            math.sqrt((params[2].grad ** 2).sum().item())
        ]

        result = optimizer_get_raw_grad_norms(optimizer)

        assert len(result) == 3
        for i, (actual, expected) in enumerate(zip(result, expected_norms)):
            assert actual == pytest.approx(expected, rel=1e-5)

    def test_consistency_with_get_grad_norm(self, multi_group_optimizer):
        """Test that raw norms compose to get_grad_norm result"""
        optimizer, params = multi_group_optimizer

        # Set gradients
        for param in params:
            param.grad = torch.randn_like(param)

        raw_norms = optimizer_get_raw_grad_norms(optimizer)
        combined_norm = optimizer_get_grad_norm(optimizer)

        # Combined norm should be L2 of raw norms
        expected_combined = math.sqrt(sum(n ** 2 for n in raw_norms))

        assert combined_norm == pytest.approx(expected_combined, rel=1e-5)

    def test_empty_param_group(self, empty_group_optimizer):
        """Test handling of empty param groups"""
        optimizer = empty_group_optimizer
        result = optimizer_get_raw_grad_norms(optimizer)

        # Should return list with one element (the empty group's norm = 0)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_all_none_gradients(self, multi_group_optimizer):
        """Test that all None gradients returns list of 0.0s"""
        optimizer, params = multi_group_optimizer
        # All gradients are None by default

        result = optimizer_get_raw_grad_norms(optimizer)

        assert len(result) == 3
        for norm in result:
            assert norm == pytest.approx(0.0, abs=1e-6)

    def test_mixed_none_and_real_gradients(self, multi_group_optimizer):
        """Test that groups with None return 0.0, others return actual norm"""
        optimizer, params = multi_group_optimizer

        # Set some gradients, leave others as None
        params[0].grad = torch.ones_like(params[0]) * 3.0
        # params[1].grad remains None
        params[2].grad = torch.ones_like(params[2]) * 5.0

        result = optimizer_get_raw_grad_norms(optimizer)

        expected_norm1 = math.sqrt((params[0].grad ** 2).sum().item())
        expected_norm2 = 0.0  # None gradients
        expected_norm3 = math.sqrt((params[2].grad ** 2).sum().item())

        assert len(result) == 3
        assert result[0] == pytest.approx(expected_norm1, rel=1e-5)
        assert result[1] == pytest.approx(expected_norm2, abs=1e-6)
        assert result[2] == pytest.approx(expected_norm3, rel=1e-5)
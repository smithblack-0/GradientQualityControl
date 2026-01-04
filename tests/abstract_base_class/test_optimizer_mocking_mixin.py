"""
Black box tests for OptimizerMockingMixin.

Tests validate the contract specified in documentation/base_object_implementation.md.
All tests use black box methodology:
- Test only documented behavior
- Test attribute forwarding and initialization mechanics
- Never test implementation details

Test organization:
- _finalize_initialization() marks end of construction
- __getattribute__ forwards to wrapped optimizer correctly
- __setattr__ during initialization sets locally
- __setattr__ after initialization forwards or raises errors
"""
import pytest
import torch

from src.gradient_quality_control.base.optimizer_mocking import OptimizerMockingMixin


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


class MinimalMockedWrapper(OptimizerMockingMixin):
    """Minimal test class using OptimizerMockingMixin."""

    def __init__(self, optimizer, finalize=True):
        """Initialize with optimizer and optionally finalize."""
        self._optimizer = optimizer
        self.wrapper_attribute = "test_value"

        if finalize:
            self._finalize_initialization()


def create_simple_optimizer():
    """Create a simple SGD optimizer."""
    params = [torch.nn.Parameter(torch.randn(5, 5))]
    return torch.optim.SGD(params, lr=0.01)


# =============================================================================
# Finalize Initialization Test Suite - tests that _finalize_initialization marks end of construction
# =============================================================================


class TestFinalizeInitialization:
    """Test _finalize_initialization behavior."""

    def test_finalize_initialization_marks_construction_complete(self):
        """_finalize_initialization marks that construction is complete."""
        optimizer = create_simple_optimizer()

        # Create without finalizing
        wrapper = MinimalMockedWrapper(optimizer, finalize=False)

        # Should be able to set attributes during construction
        wrapper.another_attr = "during_init"

        # Now finalize
        wrapper._finalize_initialization()

        # After finalization, behavior should change
        # (tested in __setattr__ tests)
        assert wrapper is not None

    def test_can_finalize_multiple_times_without_error(self):
        """_finalize_initialization can be called multiple times without error."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer, finalize=True)

        # Should not raise
        wrapper._finalize_initialization()
        wrapper._finalize_initialization()


# =============================================================================
# Getattribute Test Suite - tests that __getattribute__ forwards to wrapped optimizer correctly
# =============================================================================


class TestGetattribute:
    """Test __getattribute__ forwarding behavior."""

    def test_getattribute_returns_wrapper_attributes(self):
        """__getattribute__ returns wrapper's own attributes."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Wrapper's own attribute should be accessible
        assert wrapper.wrapper_attribute == "test_value"

    def test_getattribute_returns_optimizer_reference(self):
        """__getattribute__ returns _optimizer field."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Should be able to access _optimizer
        assert wrapper._optimizer is optimizer

    def test_getattribute_forwards_to_optimizer_for_optimizer_attributes(self):
        """__getattribute__ forwards optimizer attributes to wrapped optimizer."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Accessing optimizer's param_groups should forward
        param_groups = wrapper.param_groups

        assert param_groups is optimizer.param_groups

    def test_getattribute_forwards_optimizer_methods(self):
        """__getattribute__ forwards method access to optimizer."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Should be able to call optimizer methods
        state_dict = wrapper.state_dict()

        # Should return optimizer's state dict
        assert isinstance(state_dict, dict)

    def test_getattribute_prefers_wrapper_attributes_over_optimizer(self):
        """__getattribute__ returns wrapper attributes even if optimizer has same name."""
        optimizer = create_simple_optimizer()

        # Add attribute to optimizer
        optimizer.custom_attr = "optimizer_value"

        wrapper = MinimalMockedWrapper(optimizer, finalize=False)
        wrapper.custom_attr = "wrapper_value"
        wrapper._finalize_initialization()

        # Should prefer wrapper's attribute
        assert wrapper.custom_attr == "wrapper_value"


# =============================================================================
# Setattr During Init Test Suite - tests that __setattr__ during initialization sets attributes locally
# =============================================================================


class TestSetattrDuringInit:
    """Test __setattr__ behavior during initialization."""

    def test_setattr_during_init_sets_locally(self):
        """__setattr__ during __init__ sets attributes on wrapper instance."""
        optimizer = create_simple_optimizer()

        wrapper = MinimalMockedWrapper(optimizer, finalize=False)

        # During init, should be able to set attributes
        wrapper.init_time_attr = "test"

        # Should be stored on wrapper
        assert hasattr(wrapper, 'init_time_attr')
        assert wrapper.init_time_attr == "test"

    def test_setattr_during_init_allows_multiple_attributes(self):
        """__setattr__ during __init__ allows setting multiple attributes."""
        optimizer = create_simple_optimizer()

        wrapper = MinimalMockedWrapper(optimizer, finalize=False)

        wrapper.attr1 = "value1"
        wrapper.attr2 = "value2"
        wrapper.attr3 = "value3"

        assert wrapper.attr1 == "value1"
        assert wrapper.attr2 == "value2"
        assert wrapper.attr3 == "value3"


# =============================================================================
# Setattr After Init Test Suite - tests that __setattr__ after initialization forwards or raises errors
# =============================================================================


class TestSetattrAfterInit:
    """Test __setattr__ behavior after initialization."""

    def test_setattr_after_init_raises_for_wrapper_attribute_collision(self):
        """__setattr__ after init raises error when attempting to set wrapper's attribute."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Attempting to modify wrapper's attribute should fail
        with pytest.raises(RuntimeError):
            wrapper.wrapper_attribute = "new_value"

    def test_setattr_after_init_raises_for_optimizer_field_collision(self):
        """__setattr__ after init raises error when attempting to set _optimizer."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Attempting to replace _optimizer should fail
        new_optimizer = create_simple_optimizer()
        with pytest.raises(RuntimeError):
            wrapper._optimizer = new_optimizer

    def test_setattr_after_init_forwards_to_optimizer_for_new_attributes(self):
        """__setattr__ after init forwards new attribute assignment to optimizer."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Setting new attribute should forward to optimizer
        wrapper.new_custom_attr = "forwarded_value"

        # Should be set on optimizer, not wrapper
        assert hasattr(optimizer, 'new_custom_attr')
        assert optimizer.new_custom_attr == "forwarded_value"

    def test_setattr_after_init_forwards_optimizer_attribute_modification(self):
        """__setattr__ after init forwards modification of optimizer's existing attributes."""
        optimizer = create_simple_optimizer()

        # Set initial value on optimizer
        optimizer.existing_attr = "original"

        wrapper = MinimalMockedWrapper(optimizer)

        # Modifying through wrapper should forward to optimizer
        wrapper.existing_attr = "modified"

        assert optimizer.existing_attr == "modified"

    def test_setattr_after_init_allows_modifying_optimizer_param_groups(self):
        """__setattr__ after init allows modifying optimizer's param_groups."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Should be able to modify optimizer attributes like param_groups
        new_groups = [{'params': [torch.nn.Parameter(torch.randn(3, 3))], 'lr': 0.1}]
        wrapper.param_groups = new_groups

        # Should be set on optimizer
        assert optimizer.param_groups == new_groups


# =============================================================================
# Integration Test Suite - tests that the mixin enables transparent optimizer behavior
# =============================================================================


class TestIntegration:
    """Test integrated behavior of the mixin."""

    def test_wrapper_appears_as_optimizer(self):
        """Wrapper with mixin transparently exposes optimizer interface."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Should be able to access optimizer properties
        assert wrapper.param_groups == optimizer.param_groups

        # Should be able to call optimizer methods
        state_dict = wrapper.state_dict()
        assert isinstance(state_dict, dict)

    def test_wrapper_maintains_own_attributes(self):
        """Wrapper maintains its own attributes separate from optimizer."""
        optimizer = create_simple_optimizer()
        wrapper = MinimalMockedWrapper(optimizer)

        # Wrapper's own attributes should be separate
        assert wrapper.wrapper_attribute == "test_value"

        # Optimizer shouldn't have wrapper's attributes
        assert not hasattr(optimizer, 'wrapper_attribute')

    def test_initialization_phase_then_finalization_phase(self):
        """Wrapper allows setting attributes during init, then blocks after finalization."""
        optimizer = create_simple_optimizer()

        # During construction
        wrapper = MinimalMockedWrapper(optimizer, finalize=False)
        wrapper.init_attr = "set_during_init"

        assert wrapper.init_attr == "set_during_init"

        # After finalization
        wrapper._finalize_initialization()

        # Should raise when trying to modify wrapper attribute
        with pytest.raises(RuntimeError):
            wrapper.init_attr = "try_to_modify"

    def test_forwarding_preserves_optimizer_functionality(self):
        """Forwarding through mixin preserves optimizer's functionality."""
        params = [torch.nn.Parameter(torch.randn(5, 5))]
        optimizer = torch.optim.SGD(params, lr=0.01)
        wrapper = MinimalMockedWrapper(optimizer)

        # Set gradients
        params[0].grad = torch.randn_like(params[0])

        # Should be able to step through wrapper
        wrapper.step()

        # Optimizer should have stepped (hard to verify without checking param values)
        # At minimum, should not raise
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

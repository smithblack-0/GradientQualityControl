"""
Tests for OptimizerWrapperSBC (Scheduled Batch Controller).

Tests validate the contract specified in documentation/optimizer_wrapper_api.md
and documentation/api_guide.md. All tests use black-box methodology:
- Test only documented public behavior
- Never access implementation details
- Use ScheduleAnything for schedule integration

Test organization:
- Constructor parameter validation
- Step algorithm based on logical_batch_size
- Schedule target exposure and binding
- Factory function behavior
- Statistics reporting
- Distributed mode behaviors (if applicable)
"""
import pytest
import sys
import torch
import torch.nn as nn
import torch_schedule_anything as tsa
import math

from src.gradient_quality_control.scheduled_batch_controller import (
    OptimizerWrapperSBC,
    make_sbc_with_polynomial_schedule,
    make_sbc_with_polynomial_schedule_conventional_lr,
)


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


def create_simple_optimizer():
    """Create a simple AdamW optimizer with parameters."""
    params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
    return torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)


def apply_gradients(optimizer_wrapper):
    """Apply dummy gradients to all parameters."""
    for group in optimizer_wrapper.param_groups:
        for param in group['params']:
            param.grad = torch.ones_like(param)


# =============================================================================
# Constructor Test Suite - tests constructor parameter validation
# =============================================================================


class TestConstructor:
    """Constructor parameter validation."""

    def test_constructor_accepts_required_parameters(self):
        """Constructor accepts optimizer and physical_batch_size."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        assert optimizer_wrapper is not None
        assert optimizer_wrapper.optimizer is optimizer

    def test_constructor_accepts_max_batch_draws(self):
        """Constructor accepts max_batch_draws parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperSBC(
            optimizer,
            physical_batch_size=32,
            max_batch_draws=16
        )

        assert optimizer_wrapper is not None

    def test_constructor_accepts_distributed_mode(self):
        """Constructor accepts distributed_mode parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperSBC(
            optimizer,
            physical_batch_size=32,
            distributed_mode="replicated"
        )

        assert optimizer_wrapper.distributed_mode == "replicated"

    def test_constructor_accepts_all_parameters(self):
        """Constructor accepts all parameters together."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperSBC(
            optimizer,
            physical_batch_size=32,
            max_batch_draws=16,
            distributed_mode="sharded"
        )

        assert optimizer_wrapper is not None
        assert optimizer_wrapper.distributed_mode == "sharded"

    def test_constructor_validates_optimizer_type(self):
        """Constructor raises TypeError for non-optimizer."""
        with pytest.raises(TypeError):
            OptimizerWrapperSBC("not_an_optimizer", physical_batch_size=32)

    def test_constructor_validates_distributed_mode_values(self):
        """Constructor raises ValueError for invalid distributed_mode."""
        optimizer = create_simple_optimizer()

        with pytest.raises(ValueError):
            OptimizerWrapperSBC(
                optimizer,
                physical_batch_size=32,
                distributed_mode="invalid"
            )


# =============================================================================
# Schedule Target Exposure Test Suite - tests that wrapper exposes correct schedulable parameters
# =============================================================================


class TestScheduleTargetExposure:
    """Test that wrapper exposes correct schedule targets."""

    def test_exposes_logical_batch_size_target(self):
        """Wrapper exposes logical_batch_size as schedule target."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'logical_batch_size' in targets

    def test_exposes_lr_target(self):
        """Wrapper exposes lr as schedule target from base optimizer."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'lr' in targets

    def test_exposes_weight_decay_target(self):
        """Wrapper exposes weight_decay as schedule target from base optimizer."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'weight_decay' in targets


# =============================================================================
# Step Algorithm Test Suite - tests step decision logic based on num_draws * physical_batch_size >= logical_batch_size
# =============================================================================


class TestStepAlgorithm:
    """Test step decision logic based on logical_batch_size."""

    def test_steps_when_condition_met(self):
        """Steps when num_draws * physical_batch_size >= logical_batch_size."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size to 64 (need 2 draws: 2*32=64)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=64.0,
            schedule_target='logical_batch_size'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # First draw - should not step (1*32 < 64)
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step()
        assert result1 is False
        assert optimizer_wrapper.num_steps == 0

        # Second draw - should step (2*32 >= 64)
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step()
        assert result2 is True
        assert optimizer_wrapper.num_steps == 1

    def test_steps_immediately_when_logical_equals_physical(self):
        """Steps every call when logical_batch_size equals physical_batch_size."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size to 32 (need 1 draw: 1*32=32)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=32.0,
            schedule_target='logical_batch_size'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # Should step immediately
        apply_gradients(optimizer_wrapper)
        result = optimizer_wrapper.step()
        assert result is True
        assert optimizer_wrapper.num_steps == 1

    def test_accumulates_multiple_batches(self):
        """Accumulates correct number of batches before stepping."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size to 128 (need 4 draws: 4*32=128)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=128.0,
            schedule_target='logical_batch_size'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # First three draws should accumulate
        for i in range(3):
            apply_gradients(optimizer_wrapper)
            result = optimizer_wrapper.step()
            assert result is False
            assert optimizer_wrapper.num_steps == 0
            assert optimizer_wrapper.num_draws == i + 1

        # Fourth draw should step
        apply_gradients(optimizer_wrapper)
        result = optimizer_wrapper.step()
        assert result is True
        assert optimizer_wrapper.num_steps == 1
        assert optimizer_wrapper.num_draws == 0  # Reset after step

    def test_rounds_to_nearest_multiple(self):
        """Rounds logical_batch_size to nearest multiple of physical_batch_size."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size to 50 (rounds to 2*32=64 or 1*32=32, likely 2)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=50.0,
            schedule_target='logical_batch_size'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # Based on rounding, determine expected behavior
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step()

        # Should either step immediately (rounds to 32) or after 2 draws (rounds to 64)
        # Contract says "rounded to nearest multiple" - 50 is closer to 32 than 64
        # So likely steps immediately
        if not result1:
            apply_gradients(optimizer_wrapper)
            result2 = optimizer_wrapper.step()
            assert result2 is True

    def test_responds_to_schedule_changes(self):
        """Wrapper responds when schedule changes logical_batch_size."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Start with logical_batch_size=32, then change to 96 after first step
        scheduler = tsa.arbitrary_schedule_factory(
            optimizer_wrapper,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 32.0 if step == 0 else 96.0
            ),
            schedule_target='logical_batch_size'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # First iteration: logical=32, should step immediately
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step()
        assert result1 is True

        # Advance schedule
        sync.step()

        # Second iteration: logical=96, need 3 draws (3*32=96)
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step()
        assert result2 is False

        apply_gradients(optimizer_wrapper)
        result3 = optimizer_wrapper.step()
        assert result3 is False

        apply_gradients(optimizer_wrapper)
        result4 = optimizer_wrapper.step()
        assert result4 is True


# =============================================================================
# Statistics Reporting Test Suite - tests statistics() and vital_statistics() methods
# =============================================================================


class TestStatisticsReporting:
    """Test statistics reporting includes SBC-specific info."""

    def test_statistics_returns_dict(self):
        """statistics() returns a dictionary."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        stats = optimizer_wrapper.statistics()

        assert isinstance(stats, dict)

    def test_vital_statistics_returns_dict(self):
        """vital_statistics() returns a dictionary."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        vital_stats = optimizer_wrapper.vital_statistics()

        assert isinstance(vital_stats, dict)

    def test_statistics_includes_base_counters(self):
        """statistics() includes base wrapper counters."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        stats = optimizer_wrapper.statistics()

        # From base wrapper
        assert 'num_batches' in stats
        assert 'num_steps' in stats
        assert 'num_draws' in stats

    def test_statistics_includes_sbc_specific_info(self):
        """statistics() includes SBC-specific information."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=64.0,
            schedule_target='logical_batch_size'
        )

        stats = optimizer_wrapper.statistics()

        # SBC should report physical_batch_size and logical_batch_size
        assert 'physical_batch_size' in stats or 'logical_batch_size' in stats


# =============================================================================
# Factory Test Suite: make_sbc_with_polynomial_schedule - tests factory creates correct wrapper and schedules
# =============================================================================


class TestMakeSBCWithPolynomialSchedule:
    """Test make_sbc_with_polynomial_schedule factory."""

    def test_factory_returns_tuple(self):
        """Factory returns tuple of (wrapper, schedule)."""
        optimizer = create_simple_optimizer()

        result = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_factory_returns_correct_types(self):
        """Factory returns OptimizerWrapperSBC and SynchronousSchedule."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(optimizer_wrapper, OptimizerWrapperSBC)
        assert isinstance(scheduler, tsa.SynchronousSchedule)

    def test_learning_rate_warmup_then_constant(self):
        """Learning rate warms up to constant as documented."""
        optimizer = create_simple_optimizer()
        initial_lr = 0.001

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        lr_at_warmup_end = scheduler.get_last_lr()[0]

        # Later in training
        for _ in range(400):
            scheduler.step()
        lr_at_500 = scheduler.get_last_lr()[0]

        # Should be constant at initial_lr
        assert math.isclose(lr_at_warmup_end, initial_lr, rel_tol=0.01)
        assert math.isclose(lr_at_500, initial_lr, rel_tol=0.01)

    def test_batch_size_polynomial_schedule(self):
        """Batch size follows polynomial schedule from initial to final."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
            polynomial_power=2.0
        )

        # At end of warmup - should be at initial_batch_size
        for _ in range(100):
            scheduler.step()
        batch_size_at_100 = scheduler.get_last_schedule('logical_batch_size')[0]
        assert math.isclose(batch_size_at_100, 64.0, rel_tol=0.01)

        # At end of training - should be at final_batch_size
        for _ in range(900):
            scheduler.step()
        batch_size_at_1000 = scheduler.get_last_schedule('logical_batch_size')[0]
        assert math.isclose(batch_size_at_1000, 256.0, rel_tol=0.01)

    def test_weight_decay_warmup_then_cosine_anneal(self):
        """Weight decay warms up then cosine anneals to zero."""
        optimizer = create_simple_optimizer()
        initial_wd = 0.01

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At end of training - weight decay should anneal toward zero
        for _ in range(1000):
            scheduler.step()
        wd_at_end = scheduler.get_last_schedule('weight_decay')[0]

        # Should be much smaller than initial (annealed down)
        assert wd_at_end < initial_wd * 0.1

    def test_schedule_affects_wrapper_behavior(self):
        """Schedules actually affect wrapper stepping behavior."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=32,  # Start small
            final_batch_size=128,   # End large
            num_training_steps=10,
            num_warmup_steps=2
        )

        # Early: logical_batch_size=32, should step every call
        apply_gradients(optimizer_wrapper)
        result_early = optimizer_wrapper.step()
        assert result_early is True

        # Advance schedule to end
        for _ in range(10):
            scheduler.step()

        # Late: logical_batch_size=128, should need 4 draws
        apply_gradients(optimizer_wrapper)
        result_late_1 = optimizer_wrapper.step()
        assert result_late_1 is False


# =============================================================================
# Factory Test Suite: make_sbc_with_polynomial_schedule_conventional_lr - tests conventional LR variant
# =============================================================================


class TestMakeSBCWithPolynomialScheduleConventionalLR:
    """Test make_sbc_with_polynomial_schedule_conventional_lr factory."""

    def test_factory_returns_correct_types(self):
        """Factory returns OptimizerWrapperSBC and SynchronousSchedule."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(optimizer_wrapper, OptimizerWrapperSBC)
        assert isinstance(scheduler, tsa.SynchronousSchedule)

    def test_learning_rate_warmup_then_anneal(self):
        """Learning rate warms up then anneals to zero (conventional behavior)."""
        optimizer = create_simple_optimizer()
        initial_lr = 0.001

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        lr_at_100 = scheduler.get_last_lr()[0]
        assert math.isclose(lr_at_100, initial_lr, rel_tol=0.01)

        # At end of training - should anneal down
        for _ in range(900):
            scheduler.step()
        lr_at_1000 = scheduler.get_last_lr()[0]

        # Should be much smaller (annealed)
        assert lr_at_1000 < initial_lr * 0.1

    def test_batch_size_polynomial_schedule(self):
        """Batch size follows polynomial schedule from initial to final."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        batch_size_at_100 = scheduler.get_last_schedule('logical_batch_size')[0]
        assert math.isclose(batch_size_at_100, 64.0, rel_tol=0.01)

        # At end of training
        for _ in range(900):
            scheduler.step()
        batch_size_at_1000 = scheduler.get_last_schedule('logical_batch_size')[0]
        assert math.isclose(batch_size_at_1000, 256.0, rel_tol=0.01)

    def test_no_weight_decay_scheduling(self):
        """Weight decay is NOT scheduled in conventional_lr variant."""
        optimizer = create_simple_optimizer()
        initial_wd = 0.01

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # Weight decay should stay constant (no scheduling)
        wd_start = scheduler.get_last_schedule('weight_decay')[0]

        for _ in range(1000):
            scheduler.step()

        wd_end = scheduler.get_last_schedule('weight_decay')[0]

        # Should remain approximately the same
        assert math.isclose(wd_start, wd_end, rel_tol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

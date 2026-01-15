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
- Distributed mode behaviors
- Parameter group aggregation
"""

import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch_schedule_anything as tsa
from src.gradient_quality_control.implementations.schedule_batch_controller import (
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
        for param in group["params"]:
            param.grad = torch.ones_like(param)


def sbc_distributed_worker(
    rank,
    world_size,
    num_steps,
    physical_batch_size,
    logical_batch_size,
    distributed_mode,
    output_dir,
    master_addr,
    master_port,
):
    """
    Infrastructure worker for SBC distributed testing.

    Executes num_steps iterations, applying dummy gradients each time.
    Logs vital_statistics + stepped after each step.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        # Create wrapper
        params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
        optimizer_wrapper = OptimizerWrapperSBC(
            optimizer, physical_batch_size=physical_batch_size, distributed_mode=distributed_mode
        )

        # Bind schedule
        tsa.constant_schedule(
            optimizer_wrapper, value=logical_batch_size, schedule_target="logical_batch_size"
        )

        # Execute steps and log telemetry
        log = []
        for step_num in range(num_steps):
            for param in params:
                param.grad = torch.ones_like(param)
            result = optimizer_wrapper.step()

            stats = optimizer_wrapper.vital_statistics()
            stats["stepped"] = result
            stats["step_number"] = step_num
            log.append(stats)

        # Save log
        output_file = Path(output_dir) / f"rank_{rank}.json"
        with open(output_file, "w") as f:
            json.dump({"rank": rank, "log": log}, f)

    finally:
        dist.destroy_process_group()


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
            optimizer, physical_batch_size=32, max_batch_draws=16
        )

        assert optimizer_wrapper is not None

    def test_constructor_accepts_distributed_mode(self):
        """Constructor accepts distributed_mode parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperSBC(
            optimizer, physical_batch_size=32, distributed_mode="replicated"
        )

        assert optimizer_wrapper.distributed_mode == "replicated"

    def test_constructor_accepts_all_parameters(self):
        """Constructor accepts all parameters together."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperSBC(
            optimizer, physical_batch_size=32, max_batch_draws=16, distributed_mode="sharded"
        )

        assert optimizer_wrapper is not None
        assert optimizer_wrapper.distributed_mode == "sharded"
        assert optimizer_wrapper.max_draws == 16

    def test_constructor_validates_optimizer_type(self):
        """Constructor raises TypeError for non-optimizer."""
        with pytest.raises(TypeError):
            OptimizerWrapperSBC("not_an_optimizer", physical_batch_size=32)

    def test_constructor_validates_distributed_mode_values(self):
        """Constructor raises ValueError for invalid distributed_mode."""
        optimizer = create_simple_optimizer()

        with pytest.raises(ValueError):
            OptimizerWrapperSBC(optimizer, physical_batch_size=32, distributed_mode="invalid")

    def test_constructor_complains_when_not_given_physical_batch_size(self):
        """Constructor raises TypeError for no physical batch size."""
        optimizer = create_simple_optimizer()

        with pytest.raises(TypeError):
            OptimizerWrapperSBC(
                optimizer,
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

        assert "logical_batch_size" in targets


# =============================================================================
# Step Algorithm Test Suite - tests step decision logic based on num_draws *
# physical_batch_size >= logical_batch_size
# =============================================================================


class TestStepAlgorithm:
    """Test step decision logic based on logical_batch_size."""

    def test_steps_when_condition_met(self):
        """Steps when num_draws * physical_batch_size >= logical_batch_size."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size to 64 (need 2 draws: 2*32=64)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper, value=64.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

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
            optimizer_wrapper, value=32.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

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
            optimizer_wrapper, value=128.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

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

    def test_meets_threshold_with_exact_multiple(self):
        """Steps when num_draws * physical_batch_size >= logical_batch_size (exact)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size to 96 (exactly 3*32)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper, value=96.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

        # Need exactly 3 draws (3*32 = 96 >= 96)
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step()
        assert result1 is False

        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step()
        assert result2 is False

        apply_gradients(optimizer_wrapper)
        result3 = optimizer_wrapper.step()
        assert result3 is True

    def test_meets_threshold_with_overshoot(self):
        """Steps when num_draws * physical_batch_size >= logical_batch_size (overshoot)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size to 100 (need 4 draws: 3*32=96 < 100, 4*32=128 >= 100)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper, value=100.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

        # First 3 draws should not step
        for _ in range(3):
            apply_gradients(optimizer_wrapper)
            result = optimizer_wrapper.step()
            assert result is False

        # Fourth draw should step (4*32=128 >= 100)
        apply_gradients(optimizer_wrapper)
        result = optimizer_wrapper.step()
        assert result is True

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
            schedule_target="logical_batch_size",
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

    def test_exact_formula_accumulates_below_threshold(self):
        """Verify exact formula with step-by-step calculation - accumulation case."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set logical_batch_size
        scheduler = tsa.constant_schedule(
            optimizer_wrapper, value=100.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

        # Prediction using formula (step-by-step):
        #   Physical batch size: 32
        #   Logical batch size: 100
        #   After draw 1: effective = 1 * 32 = 32
        #   Check: 32 >= 100? NO → should NOT step
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step()
        assert result1 is False
        assert optimizer_wrapper.num_draws == 1

        # After draw 2: effective = 2 * 32 = 64
        #   Check: 64 >= 100? NO → should NOT step
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step()
        assert result2 is False
        assert optimizer_wrapper.num_draws == 2

        # After draw 3: effective = 3 * 32 = 96
        #   Check: 96 >= 100? NO → should NOT step
        apply_gradients(optimizer_wrapper)
        result3 = optimizer_wrapper.step()
        assert result3 is False
        assert optimizer_wrapper.num_draws == 3

    def test_exact_formula_steps_at_threshold(self):
        """Verify exact formula with step-by-step calculation - exact threshold case."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=25)

        # Set logical_batch_size
        scheduler = tsa.constant_schedule(
            optimizer_wrapper, value=75.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

        # Prediction using formula (step-by-step):
        #   Physical batch size: 25
        #   Logical batch size: 75
        #   After draw 1: effective = 1 * 25 = 25
        #   Check: 25 >= 75? NO → should NOT step
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step()
        assert result1 is False

        # After draw 2: effective = 2 * 25 = 50
        #   Check: 50 >= 75? NO → should NOT step
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step()
        assert result2 is False

        # After draw 3: effective = 3 * 25 = 75
        #   Check: 75 >= 75? YES → should step
        apply_gradients(optimizer_wrapper)
        result3 = optimizer_wrapper.step()
        assert result3 is True
        assert optimizer_wrapper.num_steps == 1
        assert optimizer_wrapper.num_draws == 0  # Reset after step

    def test_exact_formula_steps_with_overshoot(self):
        """Verify exact formula with step-by-step calculation - overshoot case."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=20)

        # Set logical_batch_size
        scheduler = tsa.constant_schedule(
            optimizer_wrapper, value=55.0, schedule_target="logical_batch_size"
        )
        tsa.SynchronousSchedule([scheduler])

        # Prediction using formula (step-by-step):
        #   Physical batch size: 20
        #   Logical batch size: 55
        #   After draw 1: effective = 1 * 20 = 20
        #   Check: 20 >= 55? NO → should NOT step
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step()
        assert result1 is False

        # After draw 2: effective = 2 * 20 = 40
        #   Check: 40 >= 55? NO → should NOT step
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step()
        assert result2 is False

        # After draw 3: effective = 3 * 20 = 60
        #   Check: 60 >= 55? YES → should step (overshoot by 5)
        apply_gradients(optimizer_wrapper)
        result3 = optimizer_wrapper.step()
        assert result3 is True
        assert optimizer_wrapper.num_steps == 1


# =============================================================================
# Parameter Group Aggregation Test Suite - tests MAX aggregation across multiple param groups
# =============================================================================


class TestParameterGroupAggregation:
    """Test that MAX logical_batch_size is used across parameter groups."""

    def test_uses_max_batch_size_across_groups(self):
        """Uses MAX logical_batch_size when multiple param groups have different values."""
        # Create optimizer with multiple parameter groups
        params1 = [torch.nn.Parameter(torch.randn(5, 5))]
        params2 = [torch.nn.Parameter(torch.randn(5, 5))]
        optimizer = torch.optim.AdamW(
            [{"params": params1, "lr": 0.001}, {"params": params2, "lr": 0.001}]
        )

        wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Set different logical_batch_size for each group
        # Group 0: logical_batch_size=64
        # Group 1: logical_batch_size=128
        # MAX = 128 should be used
        optimizer.param_groups[0]["logical_batch_size"] = 64.0
        optimizer.param_groups[1]["logical_batch_size"] = 128.0

        # With physical_batch_size=32, MAX=128:
        # Need 4 draws: 4*32=128 >= 128
        for i in range(3):
            for param in params1 + params2:
                param.grad = torch.ones_like(param)
            result = wrapper.step()
            assert result is False

        # Fourth draw should step
        for param in params1 + params2:
            param.grad = torch.ones_like(param)
        result = wrapper.step()
        assert result is True


# =============================================================================
# Statistics Reporting Test Suite - tests statistics() and vital_statistics() methods
# =============================================================================


class TestStatisticsReporting:
    """Test statistics reporting includes SBC-specific info."""

    def test_statistics_includes_physical_batch_size(self):
        """statistics() includes physical_batch_size (optional)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(
            optimizer,
            physical_batch_size=32,
        )

        stats = optimizer_wrapper.statistics("verbose")

        assert "physical_batch_size" in stats
        assert stats["physical_batch_size"] == 32

    def test_vital_statistics_includes_logical_batch_size(self):
        """vital_statistics() includes logical_batch_size (vital)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

        # Bind schedule to logical_batch_size
        tsa.constant_schedule(optimizer_wrapper, value=64.0, schedule_target="logical_batch_size")

        vital_stats = optimizer_wrapper.vital_statistics()

        assert "logical_batch_size" in vital_stats
        assert vital_stats["logical_batch_size"] == 64.0


# =============================================================================
# Factory Test Suite: make_sbc_with_polynomial_schedule - tests factory creates correct
# wrapper and schedules
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
            num_warmup_steps=100,
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
            num_warmup_steps=100,
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
            num_warmup_steps=100,
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
            polynomial_power=2.0,
        )

        # At end of warmup - should be at initial_batch_size
        for _ in range(100):
            scheduler.step()
        batch_size_at_100 = scheduler.get_last_schedule("logical_batch_size")[0]
        assert math.isclose(batch_size_at_100, 64.0, rel_tol=0.01)

        # At end of training - should be at final_batch_size
        for _ in range(900):
            scheduler.step()
        batch_size_at_1000 = scheduler.get_last_schedule("logical_batch_size")[0]
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
            num_warmup_steps=100,
        )

        # At end of training - weight decay should anneal toward zero
        for _ in range(1000):
            scheduler.step()
        wd_at_end = scheduler.get_last_schedule("weight_decay")[0]

        # Should be much smaller than initial (annealed down)
        assert wd_at_end < initial_wd * 0.1

    def test_schedule_affects_wrapper_behavior(self):
        """Schedules actually affect wrapper stepping behavior."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=32,  # Start small
            final_batch_size=128,  # End large
            num_training_steps=10,
            num_warmup_steps=2,
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
# Factory Test Suite: make_sbc_with_polynomial_schedule_conventional_lr - tests
# conventional LR variant
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
            num_warmup_steps=100,
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
            num_warmup_steps=100,
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
            num_warmup_steps=100,
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        batch_size_at_100 = scheduler.get_last_schedule("logical_batch_size")[0]
        assert math.isclose(batch_size_at_100, 64.0, rel_tol=0.01)

        # At end of training
        for _ in range(900):
            scheduler.step()
        batch_size_at_1000 = scheduler.get_last_schedule("logical_batch_size")[0]
        assert math.isclose(batch_size_at_1000, 256.0, rel_tol=0.01)

    def test_no_weight_decay_scheduling(self):
        """Weight decay is NOT scheduled in conventional_lr variant."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule_conventional_lr(
            optimizer=optimizer,
            physical_batch_size=32,
            initial_batch_size=64,
            final_batch_size=256,
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # Weight decay should stay constant (no scheduling)
        wd_start = scheduler.get_last_schedule("weight_decay")[0]

        for _ in range(1000):
            scheduler.step()

        wd_end = scheduler.get_last_schedule("weight_decay")[0]

        # Should remain approximately the same
        assert math.isclose(wd_start, wd_end, rel_tol=0.01)


# =============================================================================
# Distributed Mode Test Suite - tests behavioral side effects in distributed mode
# =============================================================================


class TestDistributedMode:
    """Test distributed mode behavioral side effects."""

    @pytest.mark.distributed
    @pytest.mark.skipif(sys.platform == "win32", reason="gloo not supported on Windows")
    def test_replicated_mode_multiplies_physical_batch_size(self):
        """Replicated mode multiplies physical batch size by world_size per contract."""
        world_size = 2

        # Test configuration - visible in test
        # Physical batch size: 32 per device
        # Logical batch size: 128
        # Replicated mode: effective physical = 32 * 2 = 64
        # Should step at draw 2: 2 * 64 = 128 >= 128
        num_steps = 3
        physical_batch_size = 32
        logical_batch_size = 128.0

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                sbc_distributed_worker,
                args=(
                    world_size,
                    num_steps,
                    physical_batch_size,
                    logical_batch_size,
                    "replicated",
                    tmpdir,
                    "localhost",
                    "29506",
                ),
                nprocs=world_size,
                join=True,
            )

            # Collect logs from all ranks
            logs = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    logs.append(data["log"])

            # All ranks must agree
            assert all(log == logs[0] for log in logs), "All ranks must agree"

            # Verify stepping pattern: accumulates then steps at draw 2
            assert logs[0][0]["stepped"] is False  # Step 0: 1*64 < 128
            assert logs[0][1]["stepped"] is True  # Step 1: 2*64 >= 128

    @pytest.mark.distributed
    @pytest.mark.skipif(sys.platform == "win32", reason="gloo not supported on Windows")
    def test_sharded_mode_behaves_like_non_distributed(self):
        """Sharded mode has same stepping behavior as non-distributed."""
        world_size = 2

        # Test configuration - visible in test
        # Physical batch size: 32 per device
        # Logical batch size: 64
        # Sharded mode: effective physical = 32 (no multiplication)
        # Should step at draw 2: 2 * 32 = 64 >= 64
        num_steps = 3
        physical_batch_size = 32
        logical_batch_size = 64.0

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                sbc_distributed_worker,
                args=(
                    world_size,
                    num_steps,
                    physical_batch_size,
                    logical_batch_size,
                    "sharded",
                    tmpdir,
                    "localhost",
                    "29507",
                ),
                nprocs=world_size,
                join=True,
            )

            # Collect logs from all ranks
            logs = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    logs.append(data["log"])

            # All ranks must agree
            assert all(log == logs[0] for log in logs), "All ranks must agree"

            # Verify stepping pattern: same as non-distributed
            # Step 0: 1*32 < 64, no step
            # Step 1: 2*32 >= 64, step
            assert logs[0][0]["stepped"] is False
            assert logs[0][1]["stepped"] is True

            # Compare with non-distributed to verify identical behavior
            params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
            optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
            optimizer_wrapper_normal = OptimizerWrapperSBC(optimizer, physical_batch_size=32)

            tsa.constant_schedule(
                optimizer_wrapper_normal, value=64.0, schedule_target="logical_batch_size"
            )

            for param in params:
                param.grad = torch.ones_like(param)
            result_normal_1 = optimizer_wrapper_normal.step()

            for param in params:
                param.grad = torch.ones_like(param)
            result_normal_2 = optimizer_wrapper_normal.step()

            # Sharded should match non-distributed exactly
            assert result_normal_1 is False
            assert result_normal_2 is True


# =============================================================================
# Integration Test Suite - end-to-end training with factory
# =============================================================================


class TestIntegration:
    """End-to-end integration tests with real training."""

    def test_complete_training_cycle_with_factory(self):
        """Complete training cycle using factory-created wrapper and schedules."""
        # Create simple model and data
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # Use factory to create wrapper and schedules
        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=4,
            initial_batch_size=4,
            final_batch_size=16,
            num_training_steps=100,
            num_warmup_steps=10,
            polynomial_power=2.0,
        )

        # Training loop
        for step in range(100):
            # Generate dummy batch
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))

            # Forward pass
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)

            # Backward pass
            loss.backward()

            # Step wrapper (may or may not step optimizer)
            optimizer_wrapper.step()

            # Step scheduler
            scheduler.step()

        # Verify training occurred
        assert optimizer_wrapper.num_steps > 0
        assert optimizer_wrapper.num_batches == 100

        # Verify schedules evolved
        final_batch_size = scheduler.get_last_schedule("logical_batch_size")[0]
        assert final_batch_size > 4  # Should have increased

    def test_state_dict_save_load_resume_training(self):
        """Save state mid-training, load, and resume with identical behavior."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        optimizer_wrapper, scheduler = make_sbc_with_polynomial_schedule(
            optimizer=optimizer,
            physical_batch_size=4,
            initial_batch_size=4,
            final_batch_size=16,
            num_training_steps=100,
            num_warmup_steps=10,
        )

        # Train for 50 steps
        for step in range(50):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer_wrapper.step()
            scheduler.step()

        # Save state
        wrapper_state = optimizer_wrapper.state_dict()
        scheduler_state = scheduler.state_dict()
        model_state = model.state_dict()

        steps_at_save = optimizer_wrapper.num_steps

        # Create new wrapper and resume
        model_new = nn.Linear(10, 2)
        model_new.load_state_dict(model_state)
        optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=0.001, weight_decay=0.01)

        optimizer_wrapper_new, scheduler_new = make_sbc_with_polynomial_schedule(
            optimizer=optimizer_new,
            physical_batch_size=4,
            initial_batch_size=4,
            final_batch_size=16,
            num_training_steps=100,
            num_warmup_steps=10,
        )

        optimizer_wrapper_new.load_state_dict(wrapper_state)
        scheduler_new.load_state_dict(scheduler_state)

        # Verify state restored
        assert optimizer_wrapper_new.num_steps == steps_at_save

        # Continue training
        for step in range(50):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            output = model_new(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer_wrapper_new.step()
            scheduler_new.step()

        # Verify training continued
        assert optimizer_wrapper_new.num_steps > steps_at_save


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

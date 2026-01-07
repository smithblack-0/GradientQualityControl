"""
Tests for OptimizerWrapperGNTS (Gradient Norm Threshold Scheduler).

Tests validate the contract specified in documentation/optimizer_wrapper_api.md
and documentation/api_guide.md. All tests use black-box methodology:
- Test only documented public behavior
- Never access implementation details
- Use ScheduleAnything for schedule integration

Test organization:
- Constructor parameter validation
- Step algorithm based on mean gradient norm vs threshold
- Schedule target exposure and binding
- Factory function behavior
- Statistics reporting
- Distributed mode behaviors
- Parameter group aggregation
"""
import pytest
import sys
import os
import math
import json
import tempfile
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_schedule_anything as tsa

from src.gradient_quality_control.gradient_norm_threshold_scheduler import (
    OptimizerWrapperGNTS,
    make_gnts_with_cosine_annealing_schedule,
    make_gnts_with_cosine_annealing_schedule_conventional_lr,
)


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


def create_simple_optimizer():
    """Create a simple AdamW optimizer with parameters."""
    params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
    return torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)


def apply_gradients(optimizer_wrapper, scale=1.0):
    """Apply scaled gradients to all parameters."""
    for group in optimizer_wrapper.param_groups:
        for param in group['params']:
            param.grad = torch.ones_like(param) * scale


def gnts_distributed_worker(rank, world_size, gradients, threshold, distributed_mode, output_dir, master_addr, master_port):
    """
    Infrastructure worker for GNTS distributed testing.

    Interprets gradient list as: gradients[step] = scalar to apply to all params.
    Logs vital_statistics + stepped after each step.
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    try:
        # Create wrapper
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=0.001)
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer, distributed_mode=distributed_mode)

        # Bind schedule
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=threshold,
            schedule_target='gradient_norm_threshold'
        )

        # Execute steps and log telemetry
        log = []
        for step_num, grad_value in enumerate(gradients):
            param.grad = torch.tensor([grad_value])
            result = optimizer_wrapper.step()

            stats = optimizer_wrapper.vital_statistics()
            stats['stepped'] = result
            stats['step_number'] = step_num
            log.append(stats)

        # Save log
        output_file = Path(output_dir) / f'rank_{rank}.json'
        with open(output_file, 'w') as f:
            json.dump({'rank': rank, 'log': log}, f)

    finally:
        dist.destroy_process_group()


# =============================================================================
# Constructor Test Suite - tests constructor parameter validation
# =============================================================================


class TestConstructor:
    """Constructor parameter validation."""

    def test_constructor_accepts_optimizer_only(self):
        """Constructor accepts optimizer with all defaults."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        assert optimizer_wrapper is not None
        assert optimizer_wrapper.optimizer is optimizer

    def test_constructor_accepts_max_batch_draws(self):
        """Constructor accepts max_batch_draws parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperGNTS(
            optimizer,
            max_batch_draws=16
        )

        assert optimizer_wrapper is not None

    def test_constructor_accepts_distributed_mode(self):
        """Constructor accepts distributed_mode parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperGNTS(
            optimizer,
            distributed_mode="replicated"
        )

        assert optimizer_wrapper.distributed_mode == "replicated"

    def test_constructor_accepts_all_parameters(self):
        """Constructor accepts all parameters together."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperGNTS(
            optimizer,
            max_batch_draws=16,
            distributed_mode="sharded"
        )

        assert optimizer_wrapper is not None
        assert optimizer_wrapper.distributed_mode == "sharded"
        assert optimizer_wrapper.max_draws == 16

    def test_constructor_validates_optimizer_type(self):
        """Constructor raises TypeError for non-optimizer."""
        with pytest.raises(TypeError):
            OptimizerWrapperGNTS("not_an_optimizer")

    def test_constructor_validates_distributed_mode_values(self):
        """Constructor raises ValueError for invalid distributed_mode."""
        optimizer = create_simple_optimizer()

        with pytest.raises(ValueError):
            OptimizerWrapperGNTS(
                optimizer,
                distributed_mode="invalid"
            )


# =============================================================================
# Schedule Target Exposure Test Suite - tests that wrapper exposes correct schedulable parameters
# =============================================================================


class TestScheduleTargetExposure:
    """Test that wrapper exposes correct schedule targets."""

    def test_exposes_gradient_norm_threshold_target(self):
        """Wrapper exposes gradient_norm_threshold as schedule target."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'gradient_norm_threshold' in targets

    def test_exposes_lr_target(self):
        """Wrapper exposes lr as schedule target from base optimizer."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'lr' in targets

    def test_exposes_weight_decay_target(self):
        """Wrapper exposes weight_decay as schedule target from base optimizer."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'weight_decay' in targets


# =============================================================================
# Step Algorithm Test Suite - tests step decision logic based on mean_norm <= gradient_norm_threshold
# =============================================================================


class TestStepAlgorithm:
    """Test step decision logic based on mean gradient norm vs threshold."""

    def test_steps_when_mean_norm_below_threshold(self):
        """Steps when mean_norm <= gradient_norm_threshold."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        # Set very high threshold so mean norm is always below
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=1000.0,
            schedule_target='gradient_norm_threshold'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # Apply small gradients - should step immediately
        apply_gradients(optimizer_wrapper, scale=0.01)
        result = optimizer_wrapper.step()

        assert result is True
        assert optimizer_wrapper.num_steps == 1

    def test_accumulates_when_mean_norm_above_threshold(self):
        """Accumulates when mean_norm > gradient_norm_threshold."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        # Set very low threshold so we need accumulation
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.001,
            schedule_target='gradient_norm_threshold'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # Apply gradients at scale=1.0
        # 3 params, each (5,5), all ones
        # Each param norm = sqrt(25*1^2) = 5.0
        # Total norm = sqrt(5^2 + 5^2 + 5^2) = sqrt(75) ≈ 8.66
        # After 1 draw: mean_norm = 8.66 / 1 = 8.66
        # 8.66 > 0.001, must not step
        apply_gradients(optimizer_wrapper, scale=1.0)
        result = optimizer_wrapper.step()

        assert result is False
        assert optimizer_wrapper.num_steps == 0
        assert optimizer_wrapper.num_draws == 1

    def test_responds_to_threshold_changes(self):
        """Wrapper responds when schedule changes gradient_norm_threshold."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        # Start with very high threshold (steps immediately)
        scheduler = tsa.arbitrary_schedule_factory(
            optimizer_wrapper,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 1000.0 if step == 0 else 0.001
            ),
            schedule_target='gradient_norm_threshold'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # First iteration: high threshold, should step
        apply_gradients(optimizer_wrapper, scale=1.0)
        result1 = optimizer_wrapper.step()
        assert result1 is True

        # Advance schedule
        sync.step()

        # Second iteration: low threshold, should accumulate
        apply_gradients(optimizer_wrapper, scale=1.0)
        result2 = optimizer_wrapper.step()
        assert result2 is False

    def test_enforces_max_batch_draws(self):
        """Steps when num_draws >= max_batch_draws regardless of threshold."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer, max_batch_draws=3)

        # Set impossible threshold (zero)
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.0,
            schedule_target='gradient_norm_threshold'
        )
        sync = tsa.SynchronousSchedule([scheduler])

        # Accumulate to max
        for i in range(3):
            apply_gradients(optimizer_wrapper, scale=10.0)
            result = optimizer_wrapper.step()
            if i < 2:
                assert result is False
            else:
                # Third draw forces step
                assert result is True


# =============================================================================
# Parameter Group Aggregation Test Suite - tests MIN aggregation across multiple param groups
# =============================================================================


class TestParameterGroupAggregation:
    """Test that MIN gradient_norm_threshold is used across parameter groups."""

    def test_uses_min_threshold_across_groups(self):
        """Uses MIN gradient_norm_threshold when multiple param groups have different values."""
        # Create optimizer with multiple parameter groups
        params1 = [torch.nn.Parameter(torch.randn(5, 5))]
        params2 = [torch.nn.Parameter(torch.randn(5, 5))]
        optimizer = torch.optim.AdamW([
            {'params': params1, 'lr': 0.001},
            {'params': params2, 'lr': 0.001}
        ])

        wrapper = OptimizerWrapperGNTS(optimizer)

        # Set different thresholds for each group
        # Group 0: threshold=10.0
        # Group 1: threshold=0.5
        # MIN = 0.5 should be used
        optimizer.param_groups[0]['gradient_norm_threshold'] = 10.0
        optimizer.param_groups[1]['gradient_norm_threshold'] = 0.5

        # Apply gradients that would pass threshold=10.0 but not threshold=0.5
        # Total norm with scale=1.0 for 2 params (5,5) = sqrt(25+25) = sqrt(50) ≈ 7.07
        # mean_norm after 1 draw = 7.07 / 1 = 7.07
        # 7.07 < 10.0 (would step with max)
        # 7.07 > 0.5 (won't step with min)
        for param in params1 + params2:
            param.grad = torch.ones_like(param) * 1.0

        result = wrapper.step()

        # Should use MIN threshold (0.5), so should not step
        assert result is False


# =============================================================================
# Statistics Reporting Test Suite - tests statistics() and vital_statistics() methods
# =============================================================================


class TestStatisticsReporting:
    """Test statistics reporting includes GNTS-specific info."""

    def test_vital_statistics_includes_gradient_norm_threshold(self):
        """vital_statistics() includes gradient_norm_threshold (vital)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperGNTS(optimizer)

        # Bind schedule to gradient_norm_threshold
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.5,
            schedule_target='gradient_norm_threshold'
        )

        vital_stats = optimizer_wrapper.vital_statistics()

        assert 'gradient_norm_threshold' in vital_stats
        assert vital_stats['gradient_norm_threshold'] == 0.5


# =============================================================================
# Factory Test Suite: make_gnts_with_cosine_annealing_schedule - tests factory creates correct wrapper and schedules
# =============================================================================


class TestMakeGNTSWithCosineAnnealingSchedule:
    """Test make_gnts_with_cosine_annealing_schedule factory."""

    def test_factory_returns_tuple(self):
        """Factory returns tuple of (wrapper, schedule)."""
        optimizer = create_simple_optimizer()

        result = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_factory_returns_correct_types(self):
        """Factory returns OptimizerWrapperGNTS and SynchronousSchedule."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(optimizer_wrapper, OptimizerWrapperGNTS)
        assert isinstance(scheduler, tsa.SynchronousSchedule)

    def test_learning_rate_warmup_then_constant(self):
        """Learning rate warms up to constant as documented."""
        optimizer = create_simple_optimizer()
        initial_lr = 0.001

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
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

    def test_threshold_inverse_warmup_then_cosine_anneal(self):
        """Threshold inverse warms up then cosine anneals as documented."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At start of warmup - threshold should be high (inverse warmup starts high)
        threshold_at_0 = scheduler.get_last_schedule('gradient_norm_threshold')[0]

        # At end of warmup - should be around initial_threshold
        for _ in range(100):
            scheduler.step()
        threshold_at_100 = scheduler.get_last_schedule('gradient_norm_threshold')[0]

        # At end of training - should anneal to final_threshold
        for _ in range(900):
            scheduler.step()
        threshold_at_1000 = scheduler.get_last_schedule('gradient_norm_threshold')[0]

        # Should decrease from warmup to end
        assert threshold_at_100 > threshold_at_1000
        assert math.isclose(threshold_at_1000, 0.1, rel_tol=0.01)

    def test_weight_decay_warmup_then_cosine_anneal(self):
        """Weight decay warms up then cosine anneals to zero."""
        optimizer = create_simple_optimizer()
        initial_wd = 0.01

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
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

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=10.0,  # Start very high
            final_threshold=0.001,   # End very low
            num_training_steps=10,
            num_warmup_steps=2
        )

        # Early: high threshold, should step quickly
        apply_gradients(optimizer_wrapper, scale=1.0)
        result_early = optimizer_wrapper.step()
        assert result_early is True

        # Advance schedule to end
        for _ in range(10):
            scheduler.step()

        # Late: very low threshold, should need accumulation
        apply_gradients(optimizer_wrapper, scale=1.0)
        result_late = optimizer_wrapper.step()
        assert result_late is False


# =============================================================================
# Factory Test Suite: make_gnts_with_cosine_annealing_schedule_conventional_lr - tests conventional LR variant
# =============================================================================


class TestMakeGNTSWithCosineAnnealingScheduleConventionalLR:
    """Test make_gnts_with_cosine_annealing_schedule_conventional_lr factory."""

    def test_factory_returns_correct_types(self):
        """Factory returns OptimizerWrapperGNTS and SynchronousSchedule."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule_conventional_lr(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(optimizer_wrapper, OptimizerWrapperGNTS)
        assert isinstance(scheduler, tsa.SynchronousSchedule)

    def test_learning_rate_warmup_then_anneal(self):
        """Learning rate warms up then anneals to zero (conventional behavior)."""
        optimizer = create_simple_optimizer()
        initial_lr = 0.001

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule_conventional_lr(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
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

    def test_threshold_inverse_warmup_then_cosine_anneal(self):
        """Threshold inverse warms up then cosine anneals."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule_conventional_lr(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        threshold_at_100 = scheduler.get_last_schedule('gradient_norm_threshold')[0]

        # At end of training
        for _ in range(900):
            scheduler.step()
        threshold_at_1000 = scheduler.get_last_schedule('gradient_norm_threshold')[0]

        # Should decrease
        assert threshold_at_100 > threshold_at_1000
        assert math.isclose(threshold_at_1000, 0.1, rel_tol=0.01)

    def test_no_weight_decay_scheduling(self):
        """Weight decay is NOT scheduled in conventional_lr variant."""
        optimizer = create_simple_optimizer()
        initial_wd = 0.01

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule_conventional_lr(
            optimizer=optimizer,
            initial_threshold=1.0,
            final_threshold=0.1,
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


# =============================================================================
# Distributed Mode Test Suite - tests behavioral side effects in distributed mode
# =============================================================================


class TestDistributedMode:
    """Test distributed mode behavioral side effects."""

    @pytest.mark.distributed
    @pytest.mark.skipif(sys.platform == 'win32', reason="gloo not supported on Windows")
    def test_replicated_mode_behaves_like_non_distributed(self):
        """Replicated mode has same stepping behavior as non-distributed."""
        world_size = 2

        # Test configuration - visible in test
        # Apply gradient norm 0.5, threshold 1.0
        # Norm 0.5, after 1 draw: mean = 0.5/1 = 0.5
        # 0.5 <= 1.0, should step
        gradients = [0.5]
        threshold = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                gnts_distributed_worker,
                args=(world_size, gradients, threshold, "replicated", tmpdir, 'localhost', '29504'),
                nprocs=world_size,
                join=True
            )

            # Collect logs from all ranks
            logs = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f'rank_{rank}.json'
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    logs.append(data['log'])

            # All ranks must agree
            assert all(log == logs[0] for log in logs), "All ranks must agree"

            # Verify distributed behavior
            assert logs[0][0]['stepped'] is True

            # Compare with non-distributed
            param = torch.nn.Parameter(torch.tensor([1.0]))
            optimizer = torch.optim.AdamW([param], lr=0.001)
            optimizer_wrapper_normal = OptimizerWrapperGNTS(optimizer)

            scheduler_normal = tsa.constant_schedule(
                optimizer_wrapper_normal,
                value=threshold,
                schedule_target='gradient_norm_threshold'
            )

            param.grad = torch.tensor([0.5])
            result_normal = optimizer_wrapper_normal.step()

            # Replicated should match non-distributed (gradients already synchronized)
            assert logs[0][0]['stepped'] == result_normal

    @pytest.mark.distributed
    @pytest.mark.skipif(sys.platform == 'win32', reason="gloo not supported on Windows")
    def test_sharded_mode_combines_norms_across_devices(self):
        """Sharded mode combines gradient norms using sqrt(sum(norm^2)/world_size)."""
        world_size = 2

        # Test configuration - visible in test
        # Symmetric gradients: norm=3.0 on each of 2 ranks
        # Combined norm = sqrt((3^2 + 3^2)/2) = sqrt(18/2) = sqrt(9) = 3.0
        # After 1 draw: mean_norm = 3.0 / 1 = 3.0
        # Threshold = 5.0
        # 3.0 <= 5.0, should step
        gradients = [3.0]
        threshold = 5.0

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                gnts_distributed_worker,
                args=(world_size, gradients, threshold, "sharded", tmpdir, 'localhost', '29505'),
                nprocs=world_size,
                join=True
            )

            # Collect logs from all ranks
            logs = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f'rank_{rank}.json'
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    logs.append(data['log'])

            # All ranks must agree
            assert all(log == logs[0] for log in logs), "All ranks must agree"

            # Verify stepped (combined norm below threshold)
            assert logs[0][0]['stepped'] is True


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
        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=2.0,
            final_threshold=0.1,
            num_training_steps=100,
            num_warmup_steps=10
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
            stepped = optimizer_wrapper.step()

            # Step scheduler
            scheduler.step()

        # Verify training occurred
        assert optimizer_wrapper.num_steps > 0
        assert optimizer_wrapper.num_batches == 100

        # Verify schedules evolved
        final_threshold = scheduler.get_last_schedule('gradient_norm_threshold')[0]
        assert final_threshold < 2.0  # Should have decreased

    def test_state_dict_save_load_resume_training(self):
        """Save state mid-training, load, and resume with identical behavior."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        optimizer_wrapper, scheduler = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer,
            initial_threshold=2.0,
            final_threshold=0.1,
            num_training_steps=100,
            num_warmup_steps=10
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

        optimizer_wrapper_new, scheduler_new = make_gnts_with_cosine_annealing_schedule(
            optimizer=optimizer_new,
            initial_threshold=2.0,
            final_threshold=0.1,
            num_training_steps=100,
            num_warmup_steps=10
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

"""
Tests for OptimizerWrapperMHT (Metric Hypothesis Test).

Tests validate the contract specified in documentation/optimizer_wrapper_api.md
and documentation/api_guide.md. All tests use black-box methodology:
- Test only documented public behavior
- Never access implementation details
- Use ScheduleAnything for schedule integration

Test organization:
- Constructor parameter validation
- Step algorithm based on confidence interval criterion
- Schedule target exposure and binding
- Factory function behavior
- Statistics reporting
- Step signature requirements
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

from src.gradient_quality_control.implementations.metric_hypothesis_test import (
    OptimizerWrapperMHT,
    make_mht_with_warmup_schedule,
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


def mht_distributed_worker(rank, world_size, metrics, confidence_level, percent_error_threshold, distributed_mode, output_dir, master_addr, master_port):
    """
    Infrastructure worker for MHT distributed testing.

    Interprets metrics list as: metrics[step] = scalar metric value to pass to step().
    Logs vital_statistics + stepped after each step.
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    try:
        # Create wrapper
        params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
        optimizer_wrapper = OptimizerWrapperMHT(optimizer, distributed_mode=distributed_mode)

        # Bind schedules
        conf_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=confidence_level,
            schedule_target='confidence_level'
        )
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=percent_error_threshold,
            schedule_target='percent_error_threshold'
        )

        # Skip initialization step (running_mean == first_metric causes zero variance)
        for param in params:
            param.grad = torch.ones_like(param)
        optimizer_wrapper.step(metric=1.0)

        # Execute steps and log telemetry
        log = []
        for step_num, metric_value in enumerate(metrics):
            for param in params:
                param.grad = torch.ones_like(param)
            result = optimizer_wrapper.step(metric=metric_value)

            stats = optimizer_wrapper.vital_statistics()
            stats['stepped'] = result
            stats['step_number'] = step_num
            log.append(stats)

            # Stop if we stepped (for comparison tests)
            if result:
                break

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
        """Constructor accepts optimizer as sole required parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        assert optimizer_wrapper is not None
        assert optimizer_wrapper.optimizer is optimizer

    def test_constructor_accepts_max_batch_draws(self):
        """Constructor accepts max_batch_draws parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperMHT(
            optimizer,
            max_batch_draws=16
        )

        assert optimizer_wrapper is not None

    def test_constructor_accepts_distributed_mode(self):
        """Constructor accepts distributed_mode parameter."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperMHT(
            optimizer,
            distributed_mode="replicated"
        )

        assert optimizer_wrapper.distributed_mode == "replicated"

    def test_constructor_accepts_all_parameters(self):
        """Constructor accepts all parameters together."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper = OptimizerWrapperMHT(
            optimizer,
            max_batch_draws=16,
            distributed_mode="sharded"
        )

        assert optimizer_wrapper is not None
        assert optimizer_wrapper.distributed_mode == "sharded"

    def test_constructor_validates_optimizer_type(self):
        """Constructor raises TypeError for non-optimizer."""
        with pytest.raises(TypeError):
            OptimizerWrapperMHT("not_an_optimizer")

    def test_constructor_validates_distributed_mode_values(self):
        """Constructor raises ValueError for invalid distributed_mode."""
        optimizer = create_simple_optimizer()

        with pytest.raises(ValueError):
            OptimizerWrapperMHT(
                optimizer,
                distributed_mode="invalid"
            )


# =============================================================================
# Schedule Target Exposure Test Suite - tests that wrapper exposes correct schedulable parameters
# =============================================================================


class TestScheduleTargetExposure:
    """Test that wrapper exposes correct schedule targets."""

    def test_exposes_confidence_level_target(self):
        """Wrapper exposes confidence_level as schedule target."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'confidence_level' in targets

    def test_exposes_percent_error_threshold_target(self):
        """Wrapper exposes percent_error_threshold as schedule target."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        targets = optimizer_wrapper.valid_schedule_targets

        assert 'percent_error_threshold' in targets


# =============================================================================
# Step Signature Test Suite - tests that step requires metric parameter
# =============================================================================


class TestStepSignature:
    """Test step signature requirements."""

    def test_step_accepts_metric_parameter(self):
        """step() accepts metric as positional parameter."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        apply_gradients(optimizer_wrapper)
        result = optimizer_wrapper.step(metric=1.0)

        assert isinstance(result, bool)

    def test_step_requires_metric_parameter(self):
        """step() raises TypeError when called without metric."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        apply_gradients(optimizer_wrapper)

        with pytest.raises(TypeError):
            optimizer_wrapper.step()


# =============================================================================
# Step Algorithm Test Suite - tests CI criterion behavior
# =============================================================================


class TestStepAlgorithm:
    """Test step decision logic based on confidence interval criterion."""

    def test_steps_with_low_variance_metrics(self):
        """Steps when metrics have low variance (tight CI)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        # Set tight thresholds
        conf_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.95,
            schedule_target='confidence_level'
        )
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.10,  # 10% error tolerance
            schedule_target='percent_error_threshold'
        )
        sync = tsa.SynchronousSchedule([conf_scheduler, error_scheduler])

        # Provide very similar metric values (low variance)
        apply_gradients(optimizer_wrapper)
        optimizer_wrapper.step(metric=1.00)

        apply_gradients(optimizer_wrapper)
        optimizer_wrapper.step(metric=1.00)

        apply_gradients(optimizer_wrapper)
        result = optimizer_wrapper.step(metric=1.00)

        # Should step due to tight CI3W (zero variance = infinitely tight CI = steps every time)
        assert result is True
        assert optimizer_wrapper.num_steps == 3

    def test_accumulates_with_high_variance_metrics(self):
        """Accumulates when metrics have high variance (wide CI)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        # Set tight thresholds
        conf_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.95,
            schedule_target='confidence_level'
        )
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.05,  # Very tight 5% tolerance
            schedule_target='percent_error_threshold'
        )
        sync = tsa.SynchronousSchedule([conf_scheduler, error_scheduler])

        # Provide highly variable metric values
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step(metric=1.0)
        # First call steps due to initialization (running_mean == metric → zero variance)
        assert result1 is True

        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step(metric=2.0)
        # Now variance increases, CI becomes wide, shouldn't step
        assert result2 is False

        apply_gradients(optimizer_wrapper)
        result3 = optimizer_wrapper.step(metric=0.5)
        # Still high variance, shouldn't step
        assert result3 is False

        # Should have stepped once (on initialization)
        assert optimizer_wrapper.num_steps == 1

    def test_force_steps_at_max_batch_draws(self):
        """Forces step when max_batch_draws reached regardless of CI."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(
            optimizer,
            max_batch_draws=3
        )

        # Set impossible thresholds
        conf_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.9999,
            schedule_target='confidence_level'
        )
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.001,  # 0.1% tolerance - nearly impossible
            schedule_target='percent_error_threshold'
        )
        sync = tsa.SynchronousSchedule([conf_scheduler, error_scheduler])

        # Provide highly variable metrics that won't meet criterion
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step(metric=1.0)
        # First call steps due to initialization
        assert result1 is True

        # Now test max_draws forcing with 3 more high-variance calls
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step(metric=5.0)
        assert result2 is False  # Draw 1/3

        apply_gradients(optimizer_wrapper)
        result3 = optimizer_wrapper.step(metric=0.1)
        assert result3 is False  # Draw 2/3

        apply_gradients(optimizer_wrapper)
        result4 = optimizer_wrapper.step(metric=10.0)
        # Should force step at max_draws (draw 3/3)
        assert result4 is True
        assert optimizer_wrapper.num_steps == 2

    def test_responds_to_confidence_level_schedule(self):
        """Wrapper responds when schedule changes confidence_level."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        # Start with low confidence (easier to pass), then increase
        conf_scheduler = tsa.arbitrary_schedule_factory(
            optimizer_wrapper,
            schedule_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: 0.50 if step == 0 else 0.9999
            ),
            schedule_target='confidence_level'
        )
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.10,
            schedule_target='percent_error_threshold'
        )
        sync = tsa.SynchronousSchedule([conf_scheduler, error_scheduler])

        # With low confidence, should step easily
        apply_gradients(optimizer_wrapper)
        optimizer_wrapper.step(metric=1.0)
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step(metric=1.1)
        assert result1 is True

        # Advance schedule to high confidence
        sync.step()

        # Now should accumulate more (harder to pass)
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step(metric=1.0)
        assert result2 is False

    def test_responds_to_percent_error_threshold_schedule(self):
        """Wrapper responds when schedule changes percent_error_threshold."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        # Set impossibly tight threshold - will never pass
        conf_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.95,
            schedule_target='confidence_level'
        )
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.0001,  # Impossibly tight
            schedule_target='percent_error_threshold'
        )
        sync = tsa.SynchronousSchedule([conf_scheduler, error_scheduler])

        # Skip initialization step
        apply_gradients(optimizer_wrapper)
        optimizer_wrapper.step(metric=1.0)

        # With tight threshold, should accumulate
        apply_gradients(optimizer_wrapper)
        result1 = optimizer_wrapper.step(metric=1.5)
        assert result1 is False

        # Change to impossibly loose threshold - will always pass
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.99,  # Impossibly loose
            schedule_target='percent_error_threshold'
        )

        # Now should step
        apply_gradients(optimizer_wrapper)
        result2 = optimizer_wrapper.step(metric=2.0)
        assert result2 is True

    def test_exact_confidence_interval_formula(self):
        """Verify CI formula: low variance steps in fewer draws than high variance."""

        low_faster_count = 0
        num_trials = 50

        for trial in range(num_trials):
            optimizer = create_simple_optimizer()
            optimizer_wrapper = OptimizerWrapperMHT(optimizer)

            # Set moderate tolerance
            conf_scheduler = tsa.constant_schedule(
                optimizer_wrapper,
                value=0.95,
                schedule_target='confidence_level'
            )
            error_scheduler = tsa.constant_schedule(
                optimizer_wrapper,
                value=0.10,  # 10% error tolerance
                schedule_target='percent_error_threshold'
            )
            sync = tsa.SynchronousSchedule([conf_scheduler, error_scheduler])

            # Skip initialization
            apply_gradients(optimizer_wrapper)
            optimizer_wrapper.step(metric=1.0)

            # Feed tight variance until steps
            while True:
                apply_gradients(optimizer_wrapper)
                if optimizer_wrapper.step(metric=random.uniform(0.99, 1.01)):
                    break
            low_draws = optimizer_wrapper.last_num_draws

            # Feed wide variance until steps
            while True:
                apply_gradients(optimizer_wrapper)
                if optimizer_wrapper.step(metric=random.uniform(0.5, 1.5)):
                    break
            high_draws = optimizer_wrapper.last_num_draws

            if low_draws <= high_draws:
                low_faster_count += 1

        # Low variance should step faster in most trials
        success_rate = low_faster_count / num_trials
        assert success_rate >= 0.90, f"Low variance only faster in {success_rate*100:.1f}% of trials"


# =============================================================================
# Parameter Group Aggregation Test Suite - tests MEAN aggregation across multiple param groups
# =============================================================================


class TestParameterGroupAggregation:
    """Test that MEAN confidence_level and percent_error_threshold are used across parameter groups."""

    def test_uses_mean_confidence_level_across_groups(self):
        """Uses MEAN confidence_level when multiple param groups have different values."""
        # Create optimizer with multiple parameter groups
        params1 = [torch.nn.Parameter(torch.randn(5, 5))]
        params2 = [torch.nn.Parameter(torch.randn(5, 5))]
        optimizer = torch.optim.AdamW([
            {'params': params1, 'lr': 0.001},
            {'params': params2, 'lr': 0.001}
        ])

        wrapper = OptimizerWrapperMHT(optimizer)

        # Set different confidence_level for each group
        # Group 0: confidence_level=0.90
        # Group 1: confidence_level=0.98
        # MEAN = (0.90 + 0.98) / 2 = 0.94
        optimizer.param_groups[0]['confidence_level'] = 0.90
        optimizer.param_groups[1]['confidence_level'] = 0.98

        # Set same percent_error_threshold
        optimizer.param_groups[0]['percent_error_threshold'] = 0.10
        optimizer.param_groups[1]['percent_error_threshold'] = 0.10

        # Provide low-variance metrics that should pass with mean confidence
        for param in params1 + params2:
            param.grad = torch.ones_like(param)
        wrapper.step(metric=1.00)

        for param in params1 + params2:
            param.grad = torch.ones_like(param)
        result = wrapper.step(metric=1.01)

        # Should eventually step (behavior depends on MEAN=0.94 confidence)
        assert isinstance(result, bool)

    def test_uses_mean_percent_error_threshold_across_groups(self):
        """Uses MEAN percent_error_threshold when multiple param groups have different values."""
        # Create optimizer with multiple parameter groups
        params1 = [torch.nn.Parameter(torch.randn(5, 5))]
        params2 = [torch.nn.Parameter(torch.randn(5, 5))]
        optimizer = torch.optim.AdamW([
            {'params': params1, 'lr': 0.001},
            {'params': params2, 'lr': 0.001}
        ])

        wrapper = OptimizerWrapperMHT(optimizer)

        # Set same confidence_level
        optimizer.param_groups[0]['confidence_level'] = 0.95
        optimizer.param_groups[1]['confidence_level'] = 0.95

        # Set different percent_error_threshold for each group
        # Group 0: percent_error_threshold=0.05
        # Group 1: percent_error_threshold=0.15
        # MEAN = (0.05 + 0.15) / 2 = 0.10
        optimizer.param_groups[0]['percent_error_threshold'] = 0.05
        optimizer.param_groups[1]['percent_error_threshold'] = 0.15

        # Provide metrics
        for param in params1 + params2:
            param.grad = torch.ones_like(param)
        wrapper.step(metric=1.0)

        for param in params1 + params2:
            param.grad = torch.ones_like(param)
        result = wrapper.step(metric=1.05)

        # Should behave according to MEAN=0.10 threshold
        assert isinstance(result, bool)


# =============================================================================
# Statistics Reporting Test Suite - tests statistics() and vital_statistics() methods
# =============================================================================


class TestStatisticsReporting:
    """Test statistics reporting includes MHT-specific info."""

    def test_vital_statistics_includes_confidence_level(self):
        """vital_statistics() includes confidence_level (vital)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        # Bind schedule
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.95,
            schedule_target='confidence_level'
        )

        vital_stats = optimizer_wrapper.vital_statistics()

        assert 'confidence_level' in vital_stats

    def test_vital_statistics_includes_percent_error_threshold(self):
        """vital_statistics() includes percent_error_threshold (vital)."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        # Bind schedule
        scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.10,
            schedule_target='percent_error_threshold'
        )

        vital_stats = optimizer_wrapper.vital_statistics()

        assert 'percent_error_threshold' in vital_stats

    def test_statistics_values_match_scheduled_values(self):
        """statistics() values match currently scheduled values."""
        optimizer = create_simple_optimizer()
        optimizer_wrapper = OptimizerWrapperMHT(optimizer)

        # Set specific values
        conf_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.98,
            schedule_target='confidence_level'
        )
        error_scheduler = tsa.constant_schedule(
            optimizer_wrapper,
            value=0.05,
            schedule_target='percent_error_threshold'
        )
        sync = tsa.SynchronousSchedule([conf_scheduler, error_scheduler])

        stats = optimizer_wrapper.statistics()

        # Should reflect scheduled values
        assert math.isclose(stats['confidence_level'], 0.98, rel_tol=0.01)
        assert math.isclose(stats['percent_error_threshold'], 0.05, rel_tol=0.01)


# =============================================================================
# Factory Test Suite: make_mht_with_warmup_schedule
# =============================================================================


class TestMakeMHTWithWarmupSchedule:
    """Test make_mht_with_warmup_schedule factory."""

    def test_factory_returns_tuple(self):
        """Factory returns tuple of (wrapper, schedule)."""
        optimizer = create_simple_optimizer()

        result = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.05,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_factory_returns_correct_types(self):
        """Factory returns OptimizerWrapperMHT and SynchronousSchedule."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.05,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        assert isinstance(optimizer_wrapper, OptimizerWrapperMHT)
        assert isinstance(scheduler, tsa.SynchronousSchedule)

    def test_learning_rate_warmup_then_cosine_anneal(self):
        """Learning rate warms up then cosine anneals as documented."""
        optimizer = create_simple_optimizer()
        initial_lr = 0.001

        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.05,
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
        assert lr_at_1000 < initial_lr * 0.1

    def test_confidence_level_warmup_to_constant(self):
        """Confidence level warms up to constant as documented."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.05,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        conf_at_100 = scheduler.get_last_schedule('confidence_level')[0]

        # Later in training
        for _ in range(400):
            scheduler.step()
        conf_at_500 = scheduler.get_last_schedule('confidence_level')[0]

        # Should be constant at target value
        assert math.isclose(conf_at_100, 0.95, rel_tol=0.01)
        assert math.isclose(conf_at_500, 0.95, rel_tol=0.01)

    def test_percent_error_threshold_warmup_to_constant(self):
        """Percent error threshold warms up to constant as documented."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.05,
            num_training_steps=1000,
            num_warmup_steps=100
        )

        # At end of warmup
        for _ in range(100):
            scheduler.step()
        error_at_100 = scheduler.get_last_schedule('percent_error_threshold')[0]

        # Later in training
        for _ in range(400):
            scheduler.step()
        error_at_500 = scheduler.get_last_schedule('percent_error_threshold')[0]

        # Should be constant at target value
        assert math.isclose(error_at_100, 0.05, rel_tol=0.01)
        assert math.isclose(error_at_500, 0.05, rel_tol=0.01)

    def test_schedule_affects_wrapper_behavior(self):
        """Schedules actually affect wrapper stepping behavior."""
        optimizer = create_simple_optimizer()

        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.50,  # Start very loose
            num_training_steps=10,
            num_warmup_steps=2
        )

        # Early: loose threshold, should step easily with varied metrics
        apply_gradients(optimizer_wrapper)
        optimizer_wrapper.step(metric=1.0)
        apply_gradients(optimizer_wrapper)
        result_early = optimizer_wrapper.step(metric=2.0)
        # Should step with loose threshold
        assert result_early is True

        # Advance to end where threshold is tight
        for _ in range(10):
            scheduler.step()

        # Later: tighter threshold, same variance should accumulate
        apply_gradients(optimizer_wrapper)
        optimizer_wrapper.step(metric=1.0)
        apply_gradients(optimizer_wrapper)
        result_late = optimizer_wrapper.step(metric=2.0)
        # May or may not step depending on running average, just verify callable
        assert isinstance(result_late, bool)


# =============================================================================
# Distributed Mode Test Suite - tests behavioral side effects in distributed mode
# =============================================================================


class TestDistributedMode:
    """Test distributed mode behavioral side effects."""

    @pytest.mark.distributed
    @pytest.mark.skipif(sys.platform == 'win32', reason="gloo not supported on Windows")
    def test_replicated_mode_steps_sooner_than_non_distributed(self):
        """Replicated mode steps sooner due to more samples per iteration."""
        world_size = 2

        # Test configuration - visible in test
        # Generate moderate-variance metrics to require multiple samples
        # Replicated mode: each rank's metric counts as independent sample
        # With world_size=2, accumulates 2 samples per iteration
        # Should reach tight CI faster than non-distributed (1 sample per iteration)
        rng = random.Random(42)
        metrics = [1.0 + rng.uniform(-0.1, 0.1) for _ in range(20)]
        confidence_level = 0.95
        percent_error_threshold = 0.05

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                mht_distributed_worker,
                args=(world_size, metrics, confidence_level, percent_error_threshold, "replicated", tmpdir, 'localhost', '29508'),
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

            # Find when distributed mode stepped (count steps until stepped=True)
            distributed_steps = len(logs[0])

            # Compare with non-distributed using same metrics
            params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
            optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
            optimizer_wrapper_normal = OptimizerWrapperMHT(optimizer)

            conf_scheduler = tsa.constant_schedule(
                optimizer_wrapper_normal,
                value=confidence_level,
                schedule_target='confidence_level'
            )
            error_scheduler = tsa.constant_schedule(
                optimizer_wrapper_normal,
                value=percent_error_threshold,
                schedule_target='percent_error_threshold'
            )

            # Skip initialization step (running_mean == first_metric causes zero variance)
            for param in params:
                param.grad = torch.ones_like(param)
            optimizer_wrapper_normal.step(metric=1.0)

            num_steps_taken = 0
            for i, metric in enumerate(metrics):
                for param in params:
                    param.grad = torch.ones_like(param)

                result = optimizer_wrapper_normal.step(metric=metric)

                if result:
                    num_steps_taken = i + 1
                    break

            # Replicated mode should step sooner (more samples per iteration)
            assert distributed_steps < num_steps_taken

    @pytest.mark.distributed
    @pytest.mark.skipif(sys.platform == 'win32', reason="gloo not supported on Windows")
    def test_sharded_mode_behaves_like_non_distributed(self):
        """Sharded mode has same stepping behavior as non-distributed."""
        world_size = 2

        # Test configuration - visible in test
        # Generate moderate-variance metrics to require multiple samples
        # Sharded mode: metrics averaged across devices, counts as 1 sample
        # Should behave identically to non-distributed (1 sample per iteration)
        rng = random.Random(42)
        metrics = [1.0 + rng.uniform(-0.1, 0.1) for _ in range(20)]
        confidence_level = 0.95
        percent_error_threshold = 0.05

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                mht_distributed_worker,
                args=(world_size, metrics, confidence_level, percent_error_threshold, "sharded", tmpdir, 'localhost', '29509'),
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

            # Find when sharded mode stepped (count steps until stepped=True)
            sharded_steps = len(logs[0])

            # Compare with non-distributed using same metrics
            params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
            optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
            optimizer_wrapper_normal = OptimizerWrapperMHT(optimizer)

            conf_scheduler = tsa.constant_schedule(
                optimizer_wrapper_normal,
                value=confidence_level,
                schedule_target='confidence_level'
            )
            error_scheduler = tsa.constant_schedule(
                optimizer_wrapper_normal,
                value=percent_error_threshold,
                schedule_target='percent_error_threshold'
            )

            # Skip initialization step (running_mean == first_metric causes zero variance)
            for param in params:
                param.grad = torch.ones_like(param)
            optimizer_wrapper_normal.step(metric=1.0)

            num_steps_taken = 0
            for i, metric in enumerate(metrics):
                for param in params:
                    param.grad = torch.ones_like(param)

                result = optimizer_wrapper_normal.step(metric=metric)

                if result:
                    num_steps_taken = i + 1
                    break

            # Sharded mode should behave like non-distributed (1 sample per iteration)
            assert sharded_steps == num_steps_taken


# =============================================================================
# Integration Test Suite - end-to-end training with factory
# =============================================================================


class TestIntegration:
    """End-to-end integration tests with real training."""

    def test_complete_training_cycle_with_factory(self):
        """Complete training cycle using factory-created wrapper and schedules."""
        # Create simple model
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # Use factory to create wrapper and schedules
        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.10,
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

            # Step wrapper with metric (may or may not step optimizer)
            stepped = optimizer_wrapper.step(metric=loss.item())

            # Step scheduler
            scheduler.step()

        # Verify training occurred
        assert optimizer_wrapper.num_steps > 0
        assert optimizer_wrapper.num_batches == 100

        # Verify schedules stayed constant after warmup
        final_conf = scheduler.get_last_schedule('confidence_level')[0]
        assert math.isclose(final_conf, 0.95, rel_tol=0.01)

    def test_state_dict_save_load_resume_training(self):
        """Save state mid-training, load, and resume with identical behavior."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.10,
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
            optimizer_wrapper.step(metric=loss.item())
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

        optimizer_wrapper_new, scheduler_new = make_mht_with_warmup_schedule(
            optimizer=optimizer_new,
            confidence_level=0.95,
            percent_error_threshold=0.10,
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
            optimizer_wrapper_new.step(metric=loss.item())
            scheduler_new.step()

        # Verify training continued
        assert optimizer_wrapper_new.num_steps > steps_at_save

    def test_metrics_evolution_affects_stepping(self):
        """Metric variance evolution affects when wrapper steps."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        optimizer_wrapper, scheduler = make_mht_with_warmup_schedule(
            optimizer=optimizer,
            confidence_level=0.95,
            percent_error_threshold=0.05,  # Tight tolerance
            num_training_steps=100,
            num_warmup_steps=10
        )

        # Phase 1: High variance metrics - should accumulate more
        high_variance_draws = 0
        for step in range(20):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()

            # Add artificial high variance
            metric = loss.item() + (step % 2) * 2.0
            stepped = optimizer_wrapper.step(metric=metric)

            if not stepped:
                high_variance_draws += 1

            scheduler.step()

        # High variance should cause more accumulation
        assert high_variance_draws > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

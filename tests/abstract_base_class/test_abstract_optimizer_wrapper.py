"""
Integration tests for AbstractOptimizerWrapper.

Tests validate the contract specified in documentation/optimizer_wrapper_api.md.
All tests use black box methodology:
- Test that wrapper auto-constructs correctly
- Test end-to-end user-facing behavior
- Test parameter validation
- Test distributed synchronization across processes
- Never test implementation details

Test organization:
- Constructor auto-construction and validation
- End-to-end wrapper functionality
- Scheduler integration
- Distributed synchronization integration tests
- Device property
"""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tempfile
import os
import random
import json
from pathlib import Path

from src.gradient_quality_control.base.abstract_optimizer_wrapper import AbstractOptimizerWrapper


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


class ConcreteWrapper(AbstractOptimizerWrapper):
    """Concrete wrapper implementation for testing."""

    def step(self, *args, **kwargs):
        """Simple step implementation - step every 3 batches."""
        self._batch_received()
        should_step = self.num_draws >= 3
        if should_step:
            self._take_optimizer_step()
        return should_step


def create_simple_optimizer():
    """Create a simple SGD optimizer."""
    params = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
    return torch.optim.SGD(params, lr=0.01)


def distributed_worker(rank, world_size, distributed_mode, output_dir, master_addr, master_port):
    """Worker function for distributed testing."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    try:
        # Create optimizer and wrapper
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer, distributed_mode=distributed_mode)

        # Generate random input value for this rank
        random.seed(rank)  # Different seed per rank
        input_value = float(random.randint(0, 10))

        # Define metric reader that returns this rank's value
        def metric_reader():
            return input_value

        # Define mergers
        def normal_merger(x):
            return x

        def replicated_merger(x):
            # Average across replicated models
            tensor = torch.tensor([x], dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return (tensor / world_size).item()

        def sharded_merger(x):
            # Sum across sharded models
            tensor = torch.tensor([x], dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor.item()

        # Bind metric
        wrapper._bind_metric('test_metric', metric_reader, normal_merger, replicated_merger, sharded_merger)

        # Get metric (should synchronize across ranks)
        result = wrapper._get_metric('test_metric')

        # Save both input and result to file
        output_file = Path(output_dir) / f'rank_{rank}.json'
        with open(output_file, 'w') as f:
            json.dump({'input': input_value, 'result': result}, f)

    finally:
        dist.destroy_process_group()


# =============================================================================
# Constructor Test Suite - tests that constructor auto-constructs subsystems correctly
# =============================================================================


class TestConstructor:
    """Test constructor auto-construction and validation."""

    def test_constructor_accepts_optimizer_only(self):
        """Constructor accepts optimizer with all defaults."""
        optimizer = create_simple_optimizer()

        wrapper = ConcreteWrapper(optimizer)

        assert wrapper is not None
        assert wrapper.optimizer is optimizer

    def test_constructor_accepts_max_draws_parameter(self):
        """Constructor accepts max_draws parameter."""
        optimizer = create_simple_optimizer()

        wrapper = ConcreteWrapper(optimizer, max_draws=32)

        assert wrapper is not None

    def test_constructor_accepts_distributed_mode_parameter(self):
        """Constructor accepts distributed_mode parameter."""
        optimizer = create_simple_optimizer()

        wrapper = ConcreteWrapper(optimizer, distributed_mode='replicated')

        assert wrapper.distributed_mode == 'replicated'

    def test_constructor_accepts_all_parameters(self):
        """Constructor accepts optimizer, max_draws, and distributed_mode."""
        optimizer = create_simple_optimizer()

        wrapper = ConcreteWrapper(
            optimizer=optimizer,
            max_draws=16,
            distributed_mode='sharded'
        )

        assert wrapper is not None
        assert wrapper.distributed_mode == 'sharded'

    def test_constructor_validates_optimizer_type(self):
        """Constructor raises TypeError for non-optimizer input."""
        with pytest.raises(TypeError):
            ConcreteWrapper("not_an_optimizer")

    def test_constructor_validates_max_draws_minimum(self):
        """Constructor raises ValueError for max_draws < 1."""
        optimizer = create_simple_optimizer()

        with pytest.raises(ValueError):
            ConcreteWrapper(optimizer, max_draws=0)

        with pytest.raises(ValueError):
            ConcreteWrapper(optimizer, max_draws=-1)

    def test_constructor_validates_distributed_mode_values(self):
        """Constructor raises ValueError for invalid distributed_mode."""
        optimizer = create_simple_optimizer()

        with pytest.raises(ValueError):
            ConcreteWrapper(optimizer, distributed_mode='invalid')

    def test_constructor_accepts_none_distributed_mode(self):
        """Constructor accepts None as distributed_mode."""
        optimizer = create_simple_optimizer()

        wrapper = ConcreteWrapper(optimizer, distributed_mode=None)

        assert wrapper.distributed_mode is None


# =============================================================================
# End-to-End Functionality Test Suite - tests complete wrapper functionality
# =============================================================================


class TestEndToEndFunctionality:
    """Test end-to-end wrapper functionality."""

    def test_complete_training_cycle(self):
        """Complete training cycle with gradient accumulation works correctly."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer, max_draws=64)

        # Simulate training batches
        for i in range(5):
            # Set gradients
            for group in wrapper.param_groups:
                for param in group['params']:
                    param.grad = torch.ones_like(param)

            # Step (our implementation steps every 3 batches)
            stepped = wrapper.step()

            if i < 2:
                assert not stepped  # First 2 batches don't step
            elif i == 2:
                assert stepped  # Third batch steps
            elif i < 5:
                assert not stepped  # Next 2 batches accumulate

        # Verify final state
        assert wrapper.num_batches == 5
        assert wrapper.num_steps == 1
        assert wrapper.num_draws == 2  # 2 batches accumulated after step

    def test_state_serialization_round_trip(self):
        """State dict round-trip preserves all state correctly."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer, max_draws=32)

        # Add custom state
        wrapper._set_state('custom_param', 0.95, 'vital')

        # Run some steps
        for _ in range(4):
            for group in wrapper.param_groups:
                for param in group['params']:
                    param.grad = torch.ones_like(param)
            wrapper.step()

        # Get state dict
        state_dict = wrapper.state_dict()

        # Create new wrapper and restore
        new_optimizer = create_simple_optimizer()
        new_wrapper = ConcreteWrapper(new_optimizer, max_draws=32)
        new_wrapper._set_state('custom_param', 0.0, 'vital')  # Initialize with different value

        new_wrapper.load_state_dict(state_dict)

        # Verify state restored
        assert new_wrapper.num_batches == wrapper.num_batches
        assert new_wrapper.num_steps == wrapper.num_steps
        assert new_wrapper.num_draws == wrapper.num_draws
        assert new_wrapper._get_state('custom_param') == 0.95

    def test_statistics_reporting(self):
        """Statistics reporting provides complete visibility."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        wrapper._set_state('threshold', 1.0, 'vital')
        wrapper._set_state('debug_info', 'test', 'optional')

        # Run a step
        for group in wrapper.param_groups:
            for param in group['params']:
                param.grad = torch.ones_like(param)

        wrapper.step()
        wrapper.step()
        wrapper.step()

        # Get statistics
        verbose_stats = wrapper.statistics(behavior='verbose')
        vital_stats = wrapper.vital_statistics()

        # Verbose should include everything
        assert 'num_batches' in verbose_stats
        assert 'threshold' in verbose_stats
        assert 'debug_info' in verbose_stats
        assert 'lr' in verbose_stats

        # Vital should exclude optional
        assert 'num_batches' in vital_stats
        assert 'threshold' in vital_stats
        assert 'debug_info' not in vital_stats
        assert 'lr' in vital_stats

    def test_max_draws_enforcement(self):
        """max_draws bound is enforced during accumulation."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer, max_draws=3)

        # Accumulate to max
        for _ in range(3):
            for group in wrapper.param_groups:
                for param in group['params']:
                    param.grad = torch.ones_like(param)
            wrapper.step()

        # Fourth attempt should raise
        for group in wrapper.param_groups:
            for param in group['params']:
                param.grad = torch.ones_like(param)

        with pytest.raises(RuntimeError):
            wrapper.step()

    def test_zero_grad_disabled(self):
        """zero_grad() is disabled and raises NotImplementedError."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        with pytest.raises(NotImplementedError):
            wrapper.zero_grad()

    def test_optimizer_transparency(self):
        """Wrapper transparently exposes optimizer interface."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        # Should access optimizer attributes
        assert wrapper.param_groups == optimizer.param_groups

        # Should access optimizer methods
        opt_state_dict = wrapper.state_dict()
        assert isinstance(opt_state_dict, dict)

        # Should be able to modify optimizer attributes
        wrapper.param_groups[0]['lr'] = 0.05
        assert optimizer.param_groups[0]['lr'] == 0.05

    def test_wrapper_extends_and_schedules_custom_params(self):
        """Wrapper can extend optimizer and schedule custom parameters via ScheduleAnything."""
        import torch_schedule_anything as tsa

        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        # Extend optimizer with custom parameter via wrapper
        wrapper._set_state('gradient_clip_threshold', 10.0, 'optimizer')

        # Verify it's in valid_schedule_targets
        targets = wrapper.valid_schedule_targets
        assert 'gradient_clip_threshold' in targets

        # Verify it's in param_groups
        assert optimizer.param_groups[0]['gradient_clip_threshold'] == 10.0

        # Create a schedule for the custom parameter
        scheduler = tsa.constant_schedule(
            optimizer,
            constant_value=0.5,  # Multiplier: 10.0 * 0.5 = 5.0
            schedule_target='gradient_clip_threshold'
        )
        scheduler = tsa.SynchronousSchedule([scheduler])

        # Step the schedule
        scheduler.step()

        # Verify the param_group value changed
        assert optimizer.param_groups[0]['gradient_clip_threshold'] == 5.0

        # Verify we can retrieve it through wrapper
        threshold_values = wrapper._get_state('gradient_clip_threshold')
        assert threshold_values == [5.0]


# =============================================================================
# Scheduler Integration Test Suite - tests ScheduleAnything integration
# =============================================================================


class TestSchedulerIntegration:
    """Test ScheduleAnything integration for dynamic parameter scheduling."""

    def test_valid_schedule_targets_includes_optimizer_params(self):
        """valid_schedule_targets includes native optimizer parameters."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        targets = wrapper.valid_schedule_targets

        # Should include native optimizer params that were extended
        assert isinstance(targets, list)

    def test_wrapper_extends_optimizer_with_custom_params(self):
        """Wrapper can extend optimizer with custom schedulable parameters."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        # Add custom schedulable parameter
        wrapper._set_state('threshold', 0.95, 'optimizer')

        # Should be in valid_schedule_targets
        targets = wrapper.valid_schedule_targets
        assert 'threshold' in targets

        # Should be accessible from param_groups
        assert 'threshold' in wrapper.param_groups[0]


# =============================================================================
# Distributed Synchronization Test Suite - tests distributed metric synchronization across processes
# =============================================================================


class TestDistributedSynchronization:
    """Test distributed metric synchronization across multiple processes."""

    def test_replicated_mode_averages_metrics_across_ranks(self):
        """Replicated mode averages metrics across all ranks and all ranks agree."""
        world_size = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                distributed_worker,
                args=(world_size, 'replicated', tmpdir, 'localhost', '12355'),
                nprocs=world_size,
                join=True
            )

            # Collect results from all ranks
            inputs = []
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f'rank_{rank}.json'
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    inputs.append(data['input'])
                    results.append(data['result'])

            # Verify all ranks got the same result (critical!)
            assert all(r == results[0] for r in results), "All ranks must agree on result"

            # Verify result is average of inputs
            expected = sum(inputs) / len(inputs)
            assert abs(results[0] - expected) < 1e-6, f"Expected {expected}, got {results[0]}"

    def test_sharded_mode_sums_metrics_across_ranks(self):
        """Sharded mode sums metrics across all ranks and all ranks agree."""
        world_size = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                distributed_worker,
                args=(world_size, 'sharded', tmpdir, 'localhost', '12356'),
                nprocs=world_size,
                join=True
            )

            # Collect results from all ranks
            inputs = []
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f'rank_{rank}.json'
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    inputs.append(data['input'])
                    results.append(data['result'])

            # Verify all ranks got the same result (critical!)
            assert all(r == results[0] for r in results), "All ranks must agree on result"

            # Verify result is sum of inputs
            expected = sum(inputs)
            assert abs(results[0] - expected) < 1e-6, f"Expected {expected}, got {results[0]}"

    def test_different_ranks_produce_consensus(self):
        """Different input values from each rank still produce consensus result."""
        world_size = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn workers
            mp.spawn(
                distributed_worker,
                args=(world_size, 'replicated', tmpdir, 'localhost', '12357'),
                nprocs=world_size,
                join=True
            )

            # Collect results from all ranks
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f'rank_{rank}.json'
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    results.append(data['result'])

            # All ranks must agree
            assert all(r == results[0] for r in results), "Consensus failed across ranks"


# =============================================================================
# Device Property Test Suite - tests device detection
# =============================================================================


class TestDeviceProperty:
    """Test device property."""

    def test_device_property_returns_parameter_device(self):
        """device property returns device of optimizer parameters."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        device = wrapper.device

        assert isinstance(device, torch.device)

    def test_device_property_reflects_cpu_parameters(self):
        """device property reflects CPU parameters."""
        optimizer = create_simple_optimizer()
        wrapper = ConcreteWrapper(optimizer)

        device = wrapper.device

        assert device.type == 'cpu'  # Should start on CPU


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

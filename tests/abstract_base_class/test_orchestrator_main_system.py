"""
Integration tests for OrchestratorMainSystem.

Tests validate the contract specified in documentation/base_object_implementation.md.
All tests use black box methodology:
- Test that orchestrator correctly coordinates subsystems
- Test property and method forwarding to subsystems
- Use real subsystem instances
- Never test implementation details

Test organization:
- Constructor and dependency injection
- Public property forwarding
- Public method forwarding
- Protected method forwarding for subclasses
- Abstract methods and error conditions
"""
import pytest
import torch

from src.gradient_quality_control.base.state_management import StateManagementSubsystem
from src.gradient_quality_control.base.distributed_metrics import DistributedMetricsManagementSubsystem
from src.gradient_quality_control.base.gradient_accumulation import GradientAccumulationStepSubsystem
from src.gradient_quality_control.base.reporting import ReportingSubsystem
from src.gradient_quality_control.base.orchestrator import OrchestratorMainSystem


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


class ConcreteOrchestrator(OrchestratorMainSystem):
    """Concrete implementation for testing - implements abstract step() method."""

    def step(self, *args, **kwargs):
        """Simple step implementation for testing."""
        self._batch_received()
        should_step = self.num_draws >= 2  # Step every 2 batches
        if should_step:
            self._take_optimizer_step()
        return should_step


def create_simple_optimizer():
    """Create a simple SGD optimizer."""
    params = [torch.nn.Parameter(torch.randn(5, 5))]
    return torch.optim.SGD(params, lr=0.01)


def create_orchestrator_with_subsystems(distributed_mode=None, max_draws=64):
    """Create orchestrator with all real subsystems wired up."""
    optimizer = create_simple_optimizer()

    state_mgr = StateManagementSubsystem(optimizer)
    distributed_metrics = DistributedMetricsManagementSubsystem(distributed_mode)
    accumulation = GradientAccumulationStepSubsystem(state_mgr, optimizer, max_draws)
    reporting = ReportingSubsystem(state_mgr)

    orchestrator = ConcreteOrchestrator(
        optimizer=optimizer,
        state_manager=state_mgr,
        distributed_metrics=distributed_metrics,
        accumulation=accumulation,
        reporting=reporting
    )

    return orchestrator, optimizer, state_mgr, distributed_metrics, accumulation, reporting


# =============================================================================
# Constructor Test Suite - tests that constructor wires subsystems correctly
# =============================================================================


class TestConstructor:
    """Test constructor and dependency injection."""

    def test_constructor_accepts_all_subsystems(self):
        """Constructor accepts all required subsystem dependencies."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        assert orchestrator is not None

    def test_constructor_stores_optimizer_reference(self):
        """Constructor stores optimizer reference accessibly."""
        orchestrator, optimizer, _, _, _, _ = create_orchestrator_with_subsystems()

        assert orchestrator.optimizer is optimizer

    def test_constructor_finalizes_initialization(self):
        """Constructor calls _finalize_initialization() to complete setup."""
        orchestrator, optimizer, _, _, _, _ = create_orchestrator_with_subsystems()

        # After construction, new attributes should forward to optimizer
        orchestrator.new_attr = "test"
        assert optimizer.new_attr == "test"


# =============================================================================
# Public Property Test Suite - tests that public properties forward to correct subsystems
# =============================================================================


class TestPublicProperties:
    """Test public property forwarding."""

    def test_optimizer_property_returns_wrapped_optimizer(self):
        """optimizer property returns the wrapped optimizer."""
        orchestrator, optimizer, _, _, _, _ = create_orchestrator_with_subsystems()

        assert orchestrator.optimizer is optimizer

    def test_num_batches_forwards_to_accumulation_subsystem(self):
        """num_batches property forwards to accumulation subsystem."""
        orchestrator, _, _, _, accumulation, _ = create_orchestrator_with_subsystems()

        # Verify coupling: both return same value
        assert orchestrator.num_batches == accumulation.num_batches

    def test_num_steps_forwards_to_accumulation_subsystem(self):
        """num_steps property forwards to accumulation subsystem."""
        orchestrator, _, _, _, accumulation, _ = create_orchestrator_with_subsystems()

        # Verify coupling: both return same value
        assert orchestrator.num_steps == accumulation.num_steps

    def test_num_draws_forwards_to_accumulation_subsystem(self):
        """num_draws property forwards to accumulation subsystem."""
        orchestrator, _, _, _, accumulation, _ = create_orchestrator_with_subsystems()

        # Verify coupling: both return same value
        assert orchestrator.num_draws == accumulation.num_draws

    def test_last_num_draws_forwards_to_accumulation_subsystem(self):
        """last_num_draws property forwards to accumulation subsystem."""
        orchestrator, _, _, _, accumulation, _ = create_orchestrator_with_subsystems()

        # Verify coupling: both return same value
        assert orchestrator.last_num_draws == accumulation.last_num_draws

    def test_last_grad_norm_forwards_to_accumulation_subsystem(self):
        """last_grad_norm property forwards to accumulation subsystem."""
        orchestrator, _, _, _, accumulation, _ = create_orchestrator_with_subsystems()

        # Verify coupling: both return same value
        assert orchestrator.last_grad_norm == accumulation.last_grad_norm

    def test_valid_schedule_targets_queries_state_manager(self):
        """valid_schedule_targets property returns optimizer-flagged state."""
        orchestrator, _, state_mgr, _, _, _ = create_orchestrator_with_subsystems()

        # Add optimizer parameter through orchestrator
        orchestrator._set_state('custom_param', 0.5, 'optimizer')

        targets = orchestrator.valid_schedule_targets

        # Verify it appears in targets
        assert 'custom_param' in targets

        # Verify coupling: state_mgr has it too
        state_list = state_mgr.show_state()
        assert ('custom_param', 'optimizer') in state_list

    def test_distributed_mode_forwards_to_distributed_metrics(self):
        """distributed_mode property forwards to distributed metrics subsystem."""
        orchestrator, _, _, distributed_metrics, _, _ = create_orchestrator_with_subsystems(distributed_mode='replicated')

        # Verify coupling: both return same value
        assert orchestrator.distributed_mode == distributed_metrics.distributed_mode

    def test_device_property_returns_optimizer_device(self):
        """device property returns device of first parameter."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        device = orchestrator.device

        assert isinstance(device, torch.device)


# =============================================================================
# Public Method Test Suite - tests that public methods forward to correct subsystems
# =============================================================================


class TestPublicMethods:
    """Test public method forwarding."""

    def test_statistics_forwards_to_reporting_subsystem(self):
        """statistics() forwards to reporting subsystem."""
        orchestrator, _, state_mgr, _, _, reporting = create_orchestrator_with_subsystems()

        # Setup through orchestrator
        orchestrator._set_state('test_value', 123, 'vital')

        # Get from orchestrator
        result = orchestrator.statistics(behavior='verbose')

        assert 'test_value' in result
        assert result['test_value'] == 123

        # Verify coupling: reporting subsystem returns same data
        reporting_result = reporting.statistics(behavior='verbose')
        assert reporting_result['test_value'] == result['test_value']

    def test_vital_statistics_forwards_to_reporting_subsystem(self):
        """vital_statistics() forwards to reporting subsystem."""
        orchestrator, _, _, _, _, reporting = create_orchestrator_with_subsystems()

        # Setup through orchestrator
        orchestrator._set_state('vital_metric', 456, 'vital')

        # Get from orchestrator
        result = orchestrator.vital_statistics()

        assert 'vital_metric' in result

        # Verify coupling: reporting subsystem returns same data
        reporting_result = reporting.vital_statistics()
        assert reporting_result['vital_metric'] == result['vital_metric']

    def test_state_dict_forwards_to_state_manager(self):
        """state_dict() forwards to state manager subsystem."""
        orchestrator, _, state_mgr, _, _, _ = create_orchestrator_with_subsystems()

        # Setup through orchestrator
        orchestrator._set_state('counter', 100, 'vital')
        orchestrator._set_state('banana', 3.0, 'optional')

        # Get state dict from orchestrator
        state_dict = orchestrator.state_dict()

        assert isinstance(state_dict, dict)
        assert 'wrapper_states' in state_dict

        # Verify coupling: state_mgr returns same state dict
        state_mgr_dict = state_mgr.state_dict()
        assert state_dict == state_mgr_dict

    def test_load_state_dict_forwards_to_state_manager(self):
        """load_state_dict() forwards to state manager subsystem."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        # Get initial state dict
        state_dict = orchestrator.state_dict()

        # Create new orchestrator and load state
        new_orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()
        new_orchestrator.load_state_dict(state_dict)

        # Should restore state
        assert new_orchestrator.state_dict() == state_dict

    def test_zero_grad_raises_not_implemented_error(self):
        """zero_grad() always raises NotImplementedError."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        with pytest.raises(NotImplementedError):
            orchestrator.zero_grad()


# =============================================================================
# Protected Method Test Suite - tests that protected methods forward to correct subsystems
# =============================================================================


class TestProtectedMethods:
    """Test protected method forwarding for subclass use."""

    def test_set_state_forwards_to_state_manager(self):
        """_set_state() forwards to state manager."""
        orchestrator, _, state_mgr, _, _, _ = create_orchestrator_with_subsystems()

        orchestrator._set_state('subclass_param', 0.99, 'vital')

        # Should be stored in state manager
        assert state_mgr.get_state('subclass_param') == 0.99

    def test_get_state_retrieves_from_state_manager_without_aggregation(self):
        """_get_state() without aggregation retrieves raw value from state manager."""
        orchestrator, _, state_mgr, _, _, _ = create_orchestrator_with_subsystems()

        # Setup through orchestrator
        orchestrator._set_state('param', [1.0, 2.0, 3.0], 'vital')

        # Get through orchestrator
        result = orchestrator._get_state('param')

        assert result == [1.0, 2.0, 3.0]

        # Verify coupling: state_mgr has same value
        assert state_mgr.get_state('param') == result

    def test_get_state_aggregates_with_reporting_when_specified(self):
        """_get_state() with aggregation uses reporting subsystem."""
        orchestrator, _, _, _, _, reporting = create_orchestrator_with_subsystems()

        # Setup through orchestrator
        orchestrator._set_state('values', [1.0, 2.0, 3.0], 'vital')

        # Get with aggregation through orchestrator
        result1 = orchestrator._get_state('values', aggregate_behavior='mean')
        result2 = orchestrator._get_state('values', aggregate_behavior='max')
        result3 = orchestrator._get_state('values', aggregate_behavior='min')

        assert result1 == 2.0
        assert result2 == 3.0
        assert result3 == 1.0

        # Verify coupling: reporting subsystem produces same aggregations
        assert reporting.aggregate_numeric_list([1.0, 2.0, 3.0], 'mean') == 2.0
        assert reporting.aggregate_numeric_list([1.0, 2.0, 3.0], 'max') == 3.0
        assert reporting.aggregate_numeric_list([1.0, 2.0, 3.0], 'min') == 1.0

    def test_bind_metric_forwards_to_distributed_metrics(self):
        """_bind_metric() forwards to distributed metrics subsystem."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        def reader():
            return 42.0

        def merger(x):
            return x

        # Should not raise
        orchestrator._bind_metric('test_metric', reader, merger, merger, merger)

    def test_bind_metric_uses_default_passthrough_for_normal_merger(self):
        """_bind_metric() uses passthrough lambda when normal_merger not provided."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems(distributed_mode=None)

        def reader():
            return 42.0

        def replicated_merger(x):
            return x * 2

        def sharded_merger(x):
            return x * 3

        # Bind with only 3 functions - should use default passthrough for normal
        orchestrator._bind_metric('test_metric', reader, replicated_merger, sharded_merger)

        # With distributed_mode=None, should use normal_merger (the default passthrough)
        result = orchestrator._get_metric('test_metric')

        # Should be 42.0 (no transformation from passthrough)
        assert result == 42.0

    def test_get_metric_forwards_to_distributed_metrics(self):
        """_get_metric() forwards to distributed metrics subsystem."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        def reader():
            return 99.0

        def merger(x):
            return x * 2

        orchestrator._bind_metric('test_metric', reader, merger, merger, merger)

        result = orchestrator._get_metric('test_metric')

        assert result == 198.0  # 99.0 * 2

    def test_batch_received_forwards_to_accumulation(self):
        """_batch_received() forwards to accumulation subsystem."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        initial_batches = orchestrator.num_batches

        orchestrator._batch_received()

        assert orchestrator.num_batches == initial_batches + 1

    def test_take_optimizer_step_forwards_to_accumulation(self):
        """_take_optimizer_step() forwards to accumulation subsystem."""
        orchestrator, optimizer, _, _, _, _ = create_orchestrator_with_subsystems()

        # Set gradients and accumulate a batch
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = torch.ones_like(param)

        orchestrator._batch_received()

        initial_steps = orchestrator.num_steps

        orchestrator._take_optimizer_step()

        assert orchestrator.num_steps == initial_steps + 1


# =============================================================================
# Integration Test Suite - tests that subsystems work together through orchestrator
# =============================================================================


class TestIntegration:
    """Test integrated behavior through orchestrator."""

    def test_complete_step_cycle_updates_all_subsystems(self):
        """Complete step cycle coordinates all subsystems correctly."""
        orchestrator, optimizer, _, _, _, _ = create_orchestrator_with_subsystems()

        # Set gradients
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = torch.ones_like(param)

        # Step should call batch_received and take_optimizer_step internally
        # Our concrete implementation steps every 2 batches
        stepped = orchestrator.step()
        assert not stepped  # First batch, shouldn't step

        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = torch.ones_like(param)

        stepped = orchestrator.step()
        assert stepped  # Second batch, should step

        # Verify counters updated
        assert orchestrator.num_batches == 2
        assert orchestrator.num_steps == 1
        assert orchestrator.num_draws == 0  # Reset after step
        assert orchestrator.last_num_draws == 2

    def test_state_round_trip_through_orchestrator(self):
        """State serialization round-trip preserves all state."""
        orchestrator, optimizer, _, _, _, _ = create_orchestrator_with_subsystems()

        # Add some state
        orchestrator._set_state('custom_value', 789, 'vital')

        # Set gradients and step
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = torch.ones_like(param)

        orchestrator.step()
        orchestrator.step()

        # Get state dict
        state_dict = orchestrator.state_dict()

        # Create new orchestrator and restore
        new_orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()
        new_orchestrator.load_state_dict(state_dict)

        # Should match
        assert new_orchestrator.num_batches == orchestrator.num_batches
        assert new_orchestrator.num_steps == orchestrator.num_steps
        assert new_orchestrator._get_state('custom_value') == 789

    def test_statistics_include_data_from_all_subsystems(self):
        """statistics() aggregates data from state manager, accumulation, and optimizer."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems()

        orchestrator._set_state('wrapper_metric', 123, 'vital')

        stats = orchestrator.statistics(behavior='verbose')

        # Should include accumulation counters
        assert 'num_batches' in stats
        assert 'num_steps' in stats

        # Should include wrapper state
        assert 'wrapper_metric' in stats

        # Should include optimizer params
        assert 'lr' in stats

    def test_distributed_mode_propagates_through_metrics(self):
        """Distributed mode set in constructor propagates to metric resolution."""
        orchestrator, _, _, _, _, _ = create_orchestrator_with_subsystems(distributed_mode='replicated')

        # Verify distributed mode accessible
        assert orchestrator.distributed_mode == 'replicated'

        # Bind a metric
        def reader():
            return 10.0

        def normal_merger(x):
            return x

        def replicated_merger(x):
            return x * 100  # Different behavior for replicated

        def sharded_merger(x):
            return x * 1000

        orchestrator._bind_metric('test', reader, replicated_merger, sharded_merger, normal_merger)

        # Get metric should use replicated merger
        result = orchestrator._get_metric('test')

        assert result == 1000.0  # 10.0 * 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

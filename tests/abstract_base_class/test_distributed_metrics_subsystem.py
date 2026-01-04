"""
Black box tests for DistributedMetricsManagementSubsystem.

Tests validate the contract specified in documentation/base_object_implementation.md.
All tests use black box methodology:
- Test only documented behavior
- Observe metric resolution through get_metric() calls
- Verify callable invocations (metric readers and mergers) via spies
- Never test implementation details

Test organization:
- Constructor validation and distributed_mode property
- bind_metric() registration and immutability
- get_metric() resolution and merger selection
- Error conditions and context-specific error messages
- Argument forwarding (*args, **kwargs)

Most distributed testing itself is handled in distributed metrics subsystem.
"""
import pytest
from typing import Any

from src.gradient_quality_control.base.distributed_metrics import DistributedMetricsManagementSubsystem


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================


def create_spy_callable(return_value=42):
    """Create a callable that tracks invocations and returns a fixed value."""
    call_log = []

    def spy(*args, **kwargs):
        call_log.append({'args': args, 'kwargs': kwargs})
        return return_value

    spy.call_log = call_log
    spy.was_called = lambda: len(call_log) > 0
    spy.call_count = lambda: len(call_log)
    spy.last_call_args = lambda: call_log[-1]['args'] if call_log else None
    spy.last_call_kwargs = lambda: call_log[-1]['kwargs'] if call_log else None

    return spy


def create_metric_reader(value=10.0):
    """Create metric reader that returns fixed value."""
    return create_spy_callable(return_value=value)


def create_merger(multiplier=1.0):
    """Create merger that multiplies input by multiplier."""
    call_log = []

    def merger(value):
        call_log.append(value)
        return value * multiplier

    merger.call_log = call_log
    merger.was_called = lambda: len(call_log) > 0
    merger.call_count = lambda: len(call_log)
    merger.last_input = lambda: call_log[-1] if call_log else None

    return merger


def create_failing_callable(exception_type=RuntimeError, message="Test error"):
    """Create callable that always raises an exception."""
    def failing(*args, **kwargs):
        raise exception_type(message)
    return failing


# =============================================================================
# Constructor Test Suite - tests that constructor validates distributed_mode correctly
# =============================================================================


class TestConstructorAndProperties:
    """Test constructor validation and distributed_mode property."""

    def test_constructor_accepts_none(self):
        """Constructor accepts distributed_mode=None."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)
        assert subsystem.distributed_mode is None

    def test_constructor_accepts_replicated(self):
        """Constructor accepts distributed_mode='replicated'."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="replicated")
        assert subsystem.distributed_mode == "replicated"

    def test_constructor_accepts_sharded(self):
        """Constructor accepts distributed_mode='sharded'."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="sharded")
        assert subsystem.distributed_mode == "sharded"

    def test_constructor_rejects_invalid_mode(self):
        """Constructor rejects invalid distributed_mode values."""
        with pytest.raises(ValueError):
            DistributedMetricsManagementSubsystem(distributed_state="invalid")

    def test_distributed_mode_property_readable(self):
        """distributed_mode property is readable."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="replicated")
        mode = subsystem.distributed_mode
        assert mode == "replicated"

    def test_distributed_mode_is_immutable(self):
        """distributed_mode cannot be changed after construction."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="replicated")

        # Attempting to modify should fail (either AttributeError or value stays same)
        original_mode = subsystem.distributed_mode
        try:
            subsystem.distributed_mode = "sharded"
        except AttributeError:
            pass  # Expected: property is read-only

        # Mode should still be original value
        assert subsystem.distributed_mode == original_mode


# =============================================================================
# Bind Metric Test Suite - tests that bind_metric() registers metrics correctly
# =============================================================================


class TestBindMetricRegistration:
    """Test bind_metric() metric registration."""

    def test_bind_metric_accepts_all_required_callables(self):
        """bind_metric() accepts all required callables without error."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader()
        normal = create_merger()
        replicated = create_merger()
        sharded = create_merger()

        # Should not raise
        subsystem.bind_metric("test_metric", reader, replicated, sharded, normal)

    def test_bind_metric_allows_multiple_different_metrics(self):
        """bind_metric() allows registering multiple different metrics."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        subsystem.bind_metric("metric1", create_metric_reader(), create_merger(),
                             create_merger(), create_merger())
        subsystem.bind_metric("metric2", create_metric_reader(), create_merger(),
                             create_merger(), create_merger())
        subsystem.bind_metric("metric3", create_metric_reader(), create_merger(),
                             create_merger(), create_merger())

        # All should be retrievable
        subsystem.get_metric("metric1")
        subsystem.get_metric("metric2")
        subsystem.get_metric("metric3")

    def test_bind_metric_throws_on_duplicate_name(self):
        """bind_metric() throws error when name already registered."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        subsystem.bind_metric("duplicate", create_metric_reader(), create_merger(),
                             create_merger(), create_merger())

        # Second registration with same name should fail
        with pytest.raises(RuntimeError):
            subsystem.bind_metric("duplicate", create_metric_reader(), create_merger(),
                                 create_merger(), create_merger())

    def test_bind_metric_validates_callables(self):
        """bind_metric() validates that provided objects are callable."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        # Passing non-callable should fail
        with pytest.raises(TypeError):
            subsystem.bind_metric("bad_metric", "not_callable", create_merger(),
                                 create_merger(), create_merger())


# =============================================================================
# Get Metric Test Suite - tests that get_metric() resolves metrics correctly
# =============================================================================


class TestGetMetricBasicResolution:
    """Test get_metric() basic metric resolution."""

    def test_get_metric_invokes_metric_reader(self):
        """get_metric() invokes the registered metric_reader."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=99.9)
        subsystem.bind_metric("test", reader, create_merger(),
                             create_merger(), create_merger())

        subsystem.get_metric("test")

        assert reader.was_called()

    def test_get_metric_returns_reader_value_with_none_mode(self):
        """get_metric() with distributed_mode=None applies normal_merger."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=100.0)
        normal = create_merger(multiplier=2.0)  # doubles the value

        subsystem.bind_metric("test", reader, create_merger(), create_merger(), normal)

        result = subsystem.get_metric("test")

        assert normal.was_called()
        assert result == 200.0  # 100.0 * 2.0

    def test_get_metric_can_be_called_multiple_times(self):
        """get_metric() can be called multiple times for same metric."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=42.0)
        subsystem.bind_metric("test", reader, lambda x: x, lambda x: x, lambda x: x)

        result1 = subsystem.get_metric("test")
        result2 = subsystem.get_metric("test")
        result3 = subsystem.get_metric("test")

        assert result1 == 42.0
        assert result2 == 42.0
        assert result3 == 42.0
        assert reader.call_count() == 3


# =============================================================================
# Get Metric Test Suite - tests that get_metric() selects correct merger based on distributed_mode
# =============================================================================


class TestGetMetricMergerSelection:
    """Test get_metric() selects correct merger based on distributed_mode."""

    def test_get_metric_uses_normal_merger_when_mode_none(self):
        """get_metric() uses normal_merger when distributed_mode is None."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=10.0)
        normal = create_merger(multiplier=5.0)
        replicated = create_merger(multiplier=100.0)
        sharded = create_merger(multiplier=1000.0)

        subsystem.bind_metric("test", reader, replicated, sharded, normal)
        result = subsystem.get_metric("test")

        # Should use normal merger (5x)
        assert normal.was_called()
        assert not replicated.was_called()
        assert not sharded.was_called()
        assert result == 50.0  # 10.0 * 5.0

    def test_get_metric_uses_replicated_merger_when_mode_replicated(self):
        """get_metric() uses replicated_merger when distributed_mode is 'replicated'."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="replicated")

        reader = create_metric_reader(value=10.0)
        normal = create_merger(multiplier=5.0)
        replicated = create_merger(multiplier=100.0)
        sharded = create_merger(multiplier=1000.0)

        subsystem.bind_metric("test", reader, replicated, sharded, normal)
        result = subsystem.get_metric("test")

        # Should use replicated merger (100x)
        assert not normal.was_called()
        assert replicated.was_called()
        assert not sharded.was_called()
        assert result == 1000.0  # 10.0 * 100.0

    def test_get_metric_uses_sharded_merger_when_mode_sharded(self):
        """get_metric() uses sharded_merger when distributed_mode is 'sharded'."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="sharded")

        reader = create_metric_reader(value=10.0)
        normal = create_merger(multiplier=5.0)
        replicated = create_merger(multiplier=100.0)
        sharded = create_merger(multiplier=1000.0)

        subsystem.bind_metric("test", reader, replicated, sharded, normal)
        result = subsystem.get_metric("test")

        # Should use sharded merger (1000x)
        assert not normal.was_called()
        assert not replicated.was_called()
        assert sharded.was_called()
        assert result == 10000.0  # 10.0 * 1000.0

    def test_get_metric_passes_reader_output_to_merger(self):
        """get_metric() passes metric_reader output to merger."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=123.456)
        normal = create_merger()

        subsystem.bind_metric("test", reader, create_merger(), create_merger(), normal)
        subsystem.get_metric("test")

        # Normal merger should have received reader's output
        assert normal.last_input() == 123.456


# =============================================================================
# Get Metric Test Suite - tests that get_metric() forwards arguments to metric_reader
# =============================================================================


class TestGetMetricArgumentForwarding:
    """Test get_metric() forwards *args and **kwargs to metric_reader."""

    def test_get_metric_forwards_positional_args(self):
        """get_metric() forwards positional arguments to metric_reader."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=1.0)
        subsystem.bind_metric("test", reader, lambda x: x, lambda x: x, lambda x: x)

        subsystem.get_metric("test", 10, 20, 30)

        assert reader.last_call_args() == (10, 20, 30)

    def test_get_metric_forwards_keyword_args(self):
        """get_metric() forwards keyword arguments to metric_reader."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=1.0)
        subsystem.bind_metric("test", reader, lambda x: x, lambda x: x, lambda x: x)

        subsystem.get_metric("test", foo="bar", baz=42)

        kwargs = reader.last_call_kwargs()
        assert kwargs["foo"] == "bar"
        assert kwargs["baz"] == 42

    def test_get_metric_forwards_mixed_args(self):
        """get_metric() forwards both positional and keyword arguments."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=1.0)
        subsystem.bind_metric("test", reader, lambda x: x, lambda x: x, lambda x: x)

        subsystem.get_metric("test", 100, 200, key1="value1", key2="value2")

        assert reader.last_call_args() == (100, 200)
        kwargs = reader.last_call_kwargs()
        assert kwargs["key1"] == "value1"
        assert kwargs["key2"] == "value2"

    def test_get_metric_works_with_no_args(self):
        """get_metric() works when called with no arguments."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=99.0)
        subsystem.bind_metric("test", reader, lambda x: x, lambda x: x, lambda x: x)

        result = subsystem.get_metric("test")

        assert result == 99.0
        assert reader.last_call_args() == ()
        assert reader.last_call_kwargs() == {}


# =============================================================================
# Get Metric Test Suite - tests that get_metric() handles error conditions and provides context-specific error messages
# =============================================================================


class TestGetMetricErrors:
    """Test get_metric() error conditions and error messages."""

    def test_get_metric_throws_on_unregistered_metric(self):
        """get_metric() throws error when metric not registered."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        with pytest.raises((KeyError, RuntimeError, ValueError)):
            subsystem.get_metric("nonexistent_metric")

    def test_get_metric_error_message_indicates_metric_lookup_failure(self):
        """get_metric() error for unregistered metric mentions 'metric lookup'."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        with pytest.raises((KeyError, RuntimeError, ValueError)) as exc_info:
            subsystem.get_metric("missing")

        error_message = str(exc_info.value).lower()
        assert "metric lookup" in error_message or "lookup" in error_message

    def test_get_metric_throws_when_reader_fails(self):
        """get_metric() propagates error when metric_reader raises."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        failing_reader = create_failing_callable(RuntimeError, "Reader failed")
        subsystem.bind_metric("test", failing_reader, lambda x: x, lambda x: x, lambda x: x)

        with pytest.raises(RuntimeError):
            subsystem.get_metric("test")

    def test_get_metric_error_message_indicates_metric_read_failure(self):
        """get_metric() error for reader failure mentions 'metric read'."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        failing_reader = create_failing_callable(RuntimeError, "Test error")
        subsystem.bind_metric("test", failing_reader, lambda x: x, lambda x: x, lambda x: x)

        with pytest.raises(RuntimeError) as exc_info:
            subsystem.get_metric("test")

        error_message = str(exc_info.value).lower()
        assert "metric read" in error_message or "read" in error_message

    def test_get_metric_error_indicates_none_context(self):
        """get_metric() error with mode=None indicates default/none context."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        failing_reader = create_failing_callable(RuntimeError, "Test")
        subsystem.bind_metric("test", failing_reader, lambda x: x, lambda x: x, lambda x: x)

        with pytest.raises(RuntimeError) as exc_info:
            subsystem.get_metric("test")

        error_message = str(exc_info.value).lower()
        # Should mention default/none/normal context
        assert any(word in error_message for word in ["default", "none", "normal", "non-distributed"])

    def test_get_metric_error_indicates_replicated_context(self):
        """get_metric() error with mode='replicated' indicates replicated context."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="replicated")

        failing_reader = create_failing_callable(RuntimeError, "Test")
        subsystem.bind_metric("test", failing_reader, lambda x: x, lambda x: x, lambda x: x)

        with pytest.raises(RuntimeError) as exc_info:
            subsystem.get_metric("test")

        error_message = str(exc_info.value).lower()
        assert "replicated" in error_message

    def test_get_metric_error_indicates_sharded_context(self):
        """get_metric() error with mode='sharded' indicates sharded context."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="sharded")

        failing_reader = create_failing_callable(RuntimeError, "Test")
        subsystem.bind_metric("test", failing_reader, lambda x: x, lambda x: x, lambda x: x)

        with pytest.raises(RuntimeError) as exc_info:
            subsystem.get_metric("test")

        error_message = str(exc_info.value).lower()
        assert "sharded" in error_message


# =============================================================================
# Invariants Test Suite - tests that documented invariants are maintained correctly
# =============================================================================


class TestInvariants:
    """Test documented invariants are maintained."""

    def test_metric_readers_executed_exactly_once_per_retrieval(self):
        """Metric readers are executed exactly once per get_metric() call."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state=None)

        reader = create_metric_reader(value=1.0)
        subsystem.bind_metric("test", reader, lambda x: x, lambda x: x, lambda x: x)

        subsystem.get_metric("test")
        assert reader.call_count() == 1

        subsystem.get_metric("test")
        assert reader.call_count() == 2

        subsystem.get_metric("test")
        assert reader.call_count() == 3

    def test_metric_resolution_is_read_only(self):
        """Metric resolution does not mutate subsystem state."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="replicated")

        reader = create_metric_reader(value=10.0)
        subsystem.bind_metric("test", reader, lambda x: x, lambda x: x, lambda x: x)

        # Call get_metric multiple times
        subsystem.get_metric("test")
        subsystem.get_metric("test")

        # Should still work the same way
        result = subsystem.get_metric("test")
        assert result == 10.0

        # distributed_mode should be unchanged
        assert subsystem.distributed_mode == "replicated"

    def test_distributed_mode_fixed_for_lifetime(self):
        """Distributed execution mode is fixed for lifetime of subsystem."""
        subsystem = DistributedMetricsManagementSubsystem(distributed_state="replicated")

        initial_mode = subsystem.distributed_mode

        # Bind and retrieve metrics
        subsystem.bind_metric("m1", create_metric_reader(), lambda x: x, lambda x: x, lambda x: x)
        subsystem.get_metric("m1")
        subsystem.bind_metric("m2", create_metric_reader(), lambda x: x, lambda x: x, lambda x: x)
        subsystem.get_metric("m2")

        # Mode should be unchanged
        assert subsystem.distributed_mode == initial_mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Black-box tests for arbitrary schedule system.
Tests observable behavior and contracts, not implementation details.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR

from src.gradient_quality_control.optim_utils.arbitrary_schedules import (
    ProxyDictByLR,
    ArbitraryScheduleAdapter,
    arbitrary_schedule_factory,
    SynchronousSchedule,
    throw_errors_on_desync,
)


def create_optimizer_with_feature(feature_name, feature_value, num_groups=1):
    """Create optimizer with specified arbitrary feature already set."""
    # Each dict becomes a separate param group
    param_groups = [
        {'params': [nn.Parameter(torch.randn(10))]}
        for _ in range(num_groups)
    ]
    optimizer = SGD(param_groups, lr=0.1)

    for param_group in optimizer.param_groups:
        param_group[feature_name] = feature_value

    return optimizer
# ---------------------------------------------------------------------------
# ProxyDictByLR Contract Tests
# ---------------------------------------------------------------------------


def test_proxy_sets_backend_dict():
    """Setting proxy['lr'] updates backend dict at target key."""
    backend = {'blurble': 0.01}
    proxy = ProxyDictByLR('blurble', backend)

    proxy['lr'] = 0.001

    assert backend['blurble'] == 0.001


def test_proxy_creation_requires_existing_key():
    """ProxyDictByLR raises KeyError if target key not in backend."""
    backend = {'momentum': 0.9}

    with pytest.raises(KeyError, match="flibber"):
        ProxyDictByLR('flibber', backend)


def test_proxy_detects_desync_with_error():
    """When backend modified outside proxy, raises RuntimeError if flag is True."""
    backend = {'blurble': 0.01}
    proxy = ProxyDictByLR('blurble', backend)

    # Initial set through proxy
    proxy['lr'] = 0.001

    # Modify backend directly (desync)
    backend['blurble'] = 0.005

    # Next set should detect desync
    throw_errors_on_desync(True)
    with pytest.raises(RuntimeError, match="desynced"):
        proxy['lr'] = 0.002


def test_proxy_detects_desync_with_warning():
    """When backend modified outside proxy, warns if flag is False."""
    backend = {'blurble': 0.01}
    proxy = ProxyDictByLR('blurble', backend)

    proxy['lr'] = 0.001
    backend['blurble'] = 0.005

    throw_errors_on_desync(False)
    with pytest.warns(UserWarning, match="modifying the backend"):
        proxy['lr'] = 0.002

    # Cleanup
    throw_errors_on_desync(True)


def test_proxy_namespaces_new_keys():
    """Setting new keys creates namespaced storage."""
    backend = {'glorbnax': 0.01}
    proxy = ProxyDictByLR('glorbnax', backend)

    # Set a scheduler metadata key
    proxy['initial_lr'] = 0.01

    # Should be namespaced under the proxy name
    assert 'schedules_namespace' in backend
    assert 'glorbnax' in backend['schedules_namespace']
    assert backend['schedules_namespace']['glorbnax']['initial_lr'] == 0.01


def test_proxy_can_retrieve_namespaced_keys():
    """Getting namespaced keys works through the proxy."""
    backend = {'glorbnax': 0.01}
    proxy = ProxyDictByLR('glorbnax', backend)

    proxy['initial_lr'] = 0.01
    proxy['base_momentum'] = 0.9

    assert proxy['initial_lr'] == 0.01
    assert proxy['base_momentum'] == 0.9


def test_multiple_proxies_isolated_namespaces():
    """Multiple proxies on same backend have isolated namespaces."""
    backend = {'glorbnax': 0.01, 'flubber': 0.5}

    proxy1 = ProxyDictByLR('glorbnax', backend)
    proxy2 = ProxyDictByLR('flubber', backend)

    # Both set 'initial_lr' - should not collide
    proxy1['initial_lr'] = 0.01
    proxy2['initial_lr'] = 0.5

    # Each proxy gets its own value
    assert proxy1['initial_lr'] == 0.01
    assert proxy2['initial_lr'] == 0.5

    # Backend has both isolated
    assert backend['schedules_namespace']['glorbnax']['initial_lr'] == 0.01
    assert backend['schedules_namespace']['flubber']['initial_lr'] == 0.5


def test_proxy_protects_original_keys():
    """Cannot set to keys that existed originally (except 'lr')."""
    backend = {'glorbnax': 0.01, 'momentum': 0.9}
    proxy = ProxyDictByLR('glorbnax', backend)

    # 'momentum' existed originally, should be protected
    with pytest.raises(KeyError, match="original"):
        proxy['momentum'] = 0.95


def test_proxy_allows_lr_despite_being_original():
    """'lr' key is special - always allowed even if it existed originally."""
    backend = {'glorbnax': 0.01, 'lr': 0.1}
    proxy = ProxyDictByLR('glorbnax', backend)

    # Should proxy to glorbnax, not error
    proxy['lr'] = 0.005
    assert backend['glorbnax'] == 0.005


def test_proxy_namespace_created_lazily():
    """schedules_namespace only created when needed."""
    backend = {'glorbnax': 0.01}
    proxy = ProxyDictByLR('glorbnax', backend)

    # Just setting 'lr' shouldn't create namespace
    proxy['lr'] = 0.005
    assert 'schedules_namespace' not in backend

    # Setting metadata creates it
    proxy['initial_lr'] = 0.01
    assert 'schedules_namespace' in backend


def test_proxy_handles_missing_namespaced_key():
    """Getting a key that doesn't exist in namespace raises KeyError."""
    backend = {'glorbnax': 0.01}
    proxy = ProxyDictByLR('glorbnax', backend)

    with pytest.raises(KeyError):
        _ = proxy['nonexistent_key']


def test_proxy_lr_key_always_proxies():
    """'lr' key always proxies to target, never goes to namespace."""
    backend = {'glorbnax': 0.01}
    proxy = ProxyDictByLR('glorbnax', backend)

    proxy['lr'] = 0.005

    # Should update backend directly, not create namespace
    assert backend['glorbnax'] == 0.005
    assert 'schedules_namespace' not in backend


def test_proxy_desync_check_still_works_with_namespace():
    """Desync detection still works even with namespace features."""
    backend = {'glorbnax': 0.01}
    proxy = ProxyDictByLR('glorbnax', backend)

    # Add some namespaced data
    proxy['initial_lr'] = 0.01

    # Set and then desync the main target
    proxy['lr'] = 0.005
    backend['glorbnax'] = 0.02  # Direct backend modification

    throw_errors_on_desync(True)
    with pytest.raises(RuntimeError, match="desynced"):
        proxy['lr'] = 0.003

# ---------------------------------------------------------------------------
# ArbitraryScheduleAdapter Contract Tests
# ---------------------------------------------------------------------------


def test_adapter_forwards_to_real_optimizer():
    """When scheduler sets adapter param_groups, real optimizer changes."""
    optimizer = create_optimizer_with_feature('glorbnax', 0.01, num_groups=2)
    adapter = ArbitraryScheduleAdapter(optimizer, 'glorbnax')

    # Simulate scheduler setting values (remembering schedulers multiply base values)
    adapter.param_groups[0]['lr'] = 0.005
    adapter.param_groups[1]['lr'] = 0.02

    assert optimizer.param_groups[0]['glorbnax'] == 0.005
    assert optimizer.param_groups[1]['glorbnax'] == 0.02


def test_adapter_has_feature_name():
    """Adapter exposes feature_name attribute."""
    optimizer = create_optimizer_with_feature('glorbnax', 0.01)
    adapter = ArbitraryScheduleAdapter(optimizer, 'glorbnax')

    assert adapter.feature_name == 'glorbnax'


def test_adapter_step_raises():
    """Adapter.step() raises NotImplementedError (it's a stub)."""
    optimizer = create_optimizer_with_feature('glorbnax', 0.01)
    adapter = ArbitraryScheduleAdapter(optimizer, 'glorbnax')

    with pytest.raises(NotImplementedError, match="stub"):
        adapter.step()


def test_adapter_state_dict_raises():
    """Adapter.state_dict() raises NotImplementedError (it's a stub)."""
    optimizer = create_optimizer_with_feature('glorbnax', 0.01)
    adapter = ArbitraryScheduleAdapter(optimizer, 'glorbnax')

    with pytest.raises(NotImplementedError, match="stub"):
        adapter.state_dict()


def test_adapter_load_state_dict_raises():
    """Adapter.load_state_dict() raises NotImplementedError (it's a stub)."""
    optimizer = create_optimizer_with_feature('glorbnax', 0.01)
    adapter = ArbitraryScheduleAdapter(optimizer, 'glorbnax')

    with pytest.raises(NotImplementedError, match="stub"):
        adapter.load_state_dict({})


# ---------------------------------------------------------------------------
# arbitrary_schedule_factory Contract Tests
# ---------------------------------------------------------------------------


def test_factory_creates_working_scheduler():
    """Scheduler from factory modifies target feature in real optimizer."""
    optimizer = create_optimizer_with_feature('zippity', 0.01)

    # Create scheduler that multiplies by 0.5 (remembering PyTorch scheduler behavior)
    scheduler = arbitrary_schedule_factory(
        'zippity',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 0.5)
    )

    scheduler.step()

    # Base value 0.01 * 0.5 = 0.005
    assert optimizer.param_groups[0]['zippity'] == 0.005


def test_factory_with_default_value():
    """Factory works with default_value for non-existent features."""
    params = [nn.Parameter(torch.randn(10))]
    optimizer = SGD(params, lr=0.1)
    # Note: 'blooper' doesn't exist in optimizer yet

    scheduler = arbitrary_schedule_factory(
        'blooper',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0),
        default_value=1.0
    )

    scheduler.step()

    # Base value 1.0 * 2.0 = 2.0
    assert optimizer.param_groups[0]['blooper'] == 2.0


def test_factory_scheduler_steps_multiple_times():
    """Scheduler continues working across multiple steps."""
    optimizer = create_optimizer_with_feature('zippity', 1.0)

    # Scheduler that doubles each step
    scheduler = arbitrary_schedule_factory(
        'zippity',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0 ** epoch)
    )

    scheduler.step()  # epoch 0: 1.0 * 2^0 = 1.0
    assert optimizer.param_groups[0]['zippity'] == 2.0

    scheduler.step()  # epoch 1: 1.0 * 2^1 = 2.0
    assert optimizer.param_groups[0]['zippity'] == 2.0

    scheduler.step()  # epoch 2: 1.0 * 2^2 = 4.0
    assert optimizer.param_groups[0]['zippity'] == 4.0


def test_factory_works_with_multiple_param_groups():
    """Scheduler from factory handles multiple param groups."""
    optimizer = create_optimizer_with_feature('zippity', 0.01, num_groups=3)

    scheduler = arbitrary_schedule_factory(
        'zippity',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 0.1)
    )

    scheduler.step()

    # All param groups should be updated
    for param_group in optimizer.param_groups:
        assert param_group['zippity'] == 0.001


# ---------------------------------------------------------------------------
# SynchronousSchedule Contract Tests
# ---------------------------------------------------------------------------


def test_synchronous_schedule_steps_all_schedules():
    """Stepping SynchronousSchedule steps all contained schedules."""
    # Create optimizer with both lr and an arbitrary feature
    params = [nn.Parameter(torch.randn(10))]
    optimizer = SGD(params, lr=1.0)
    optimizer.param_groups[0]['flubber'] = 1.0

    # Create two schedules
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5)
    flubber_scheduler = arbitrary_schedule_factory(
        'flubber',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0)
    )

    sync = SynchronousSchedule([lr_scheduler, flubber_scheduler])
    sync.step()

    assert optimizer.param_groups[0]['lr'] == 0.5
    assert optimizer.param_groups[0]['flubber'] == 2.0


def test_synchronous_schedule_rejects_duplicate_names():
    """Creating SynchronousSchedule with duplicate names raises RuntimeError."""
    params = [nn.Parameter(torch.randn(10))]
    optimizer = SGD(params, lr=1.0)

    lr_scheduler1 = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5)
    lr_scheduler2 = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.8)

    with pytest.raises(RuntimeError, match="already a schedule named 'lr'"):
        SynchronousSchedule([lr_scheduler1, lr_scheduler2])


def test_synchronous_schedule_get_last_schedule():
    """get_last_schedule returns values for named schedule."""
    params = [nn.Parameter(torch.randn(10))]
    optimizer = SGD(params, lr=1.0)
    optimizer.param_groups[0]['flubber'] = 1.0

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5)
    flubber_scheduler = arbitrary_schedule_factory(
        'flubber',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0)
    )

    sync = SynchronousSchedule([lr_scheduler, flubber_scheduler])
    sync.step()

    assert sync.get_last_schedule('lr') == [0.5]
    assert sync.get_last_schedule('flubber') == [2.0]


def test_synchronous_schedule_get_last_lr():
    """get_last_lr returns lr schedule values."""
    params = [nn.Parameter(torch.randn(10))]
    optimizer = SGD(params, lr=1.0)
    optimizer.param_groups[0]['flubber'] = 1.0

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5)
    flubber_scheduler = arbitrary_schedule_factory(
        'flubber',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0)
    )

    sync = SynchronousSchedule([lr_scheduler, flubber_scheduler])
    sync.step()

    assert sync.get_last_lr() == [0.5]


def test_synchronous_schedule_names_property():
    """schedule_names property returns list of schedule names."""
    params = [nn.Parameter(torch.randn(10))]
    optimizer = SGD(params, lr=1.0)
    optimizer.param_groups[0]['flubber'] = 1.0

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5)
    flubber_scheduler = arbitrary_schedule_factory(
        'flubber',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0)
    )

    sync = SynchronousSchedule([lr_scheduler, flubber_scheduler])

    names = sync.schedule_names
    assert 'lr' in names
    assert 'flubber' in names
    assert len(names) == 2


def test_synchronous_schedule_state_dict_roundtrip():
    """State dict save/load preserves schedule state."""
    params = [nn.Parameter(torch.randn(10))]
    optimizer = SGD(params, lr=1.0)
    optimizer.param_groups[0]['flubber'] = 1.0

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
    flubber_scheduler = arbitrary_schedule_factory(
        'flubber',
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0 ** epoch)
    )

    sync1 = SynchronousSchedule([lr_scheduler, flubber_scheduler])

    # Step a few times
    sync1.step()
    sync1.step()

    # Save state
    state = sync1.state_dict()

    # Create new sync schedule and load
    params2 = [nn.Parameter(torch.randn(10))]
    optimizer2 = SGD(params2, lr=1.0)
    optimizer2.param_groups[0]['flubber'] = 1.0

    lr_scheduler2 = LambdaLR(optimizer2, lr_lambda=lambda epoch: 0.5 ** epoch)
    flubber_scheduler2 = arbitrary_schedule_factory(
        'flubber',
        optimizer2,
        lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 2.0 ** epoch)
    )

    sync2 = SynchronousSchedule([lr_scheduler2, flubber_scheduler2])
    sync2.load_state_dict(state)

    # Step both once more and compare
    sync1.step()
    sync2.step()

    assert optimizer.param_groups[0]['lr'] == optimizer2.param_groups[0]['lr']
    assert optimizer.param_groups[0]['flubber'] == optimizer2.param_groups[0]['flubber']


# ---------------------------------------------------------------------------
# Global Flag Tests
# ---------------------------------------------------------------------------


def test_throw_errors_on_desync_changes_behavior():
    """throw_errors_on_desync flag controls error vs warning behavior."""
    backend = {'blurble': 0.01}
    proxy = ProxyDictByLR('blurble', backend)
    proxy['lr'] = 0.001

    # Desync the backend
    backend['blurble'] = 0.005

    # Test with errors on
    throw_errors_on_desync(True)
    with pytest.raises(RuntimeError):
        proxy['lr'] = 0.002

    # Reset
    proxy['lr'] = 0.002
    backend['blurble'] = 0.005

    # Test with errors off
    throw_errors_on_desync(False)
    with pytest.warns(UserWarning):
        proxy['lr'] = 0.003

    # Cleanup
    throw_errors_on_desync(True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
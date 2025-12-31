# Wrapper Factories API Guide

Convenience factory functions for creating pre-configured optimizer wrapper and schedule pairs. These factories encapsulate best-practice schedule configurations for common use cases.

## Navigation

- **[User Guide](user_guide.md)** - Usage patterns and library overview
- **[Optimizer Wrapper API](optimizer_wrapper_api.md)** - Individual wrapper specifications
- **[Base Object API](base_object_api.md)** - Abstract base class for extending

---

## make_sbc_with_polynomial_schedule

Creates an OptimizerWrapperSBC with polynomial batch size schedule.

### Signature

```python
def make_sbc_with_polynomial_schedule(
    optimizer: torch.optim.Optimizer,
    physical_batch_size: int,
    initial_batch_size: int,
    final_batch_size: int,
    total_steps: int,
    warmup_steps: int,
    polynomial_power: float = 2.0,
    max_batch_draws: int = 64
) -> Tuple[OptimizerWrapperSBC, ScheduleAnything]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer (uses existing lr and weight_decay values)
- `physical_batch_size` - Size of each microbatch; required to infer logical batch size.
- `initial_batch_size` - Starting logical batch size after warmup completed. 
- `final_batch_size` - Ending logical batch size.
- `total_steps` - Total training steps for schedule duration
- `warmup_steps` - Steps for initial warmup phase; Warmup executes by aggressively stepping
- `polynomial_power` - Exponent for polynomial curve (default: 2.0 for quadratic)
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)

### Returns

Tuple of (OptimizerWrapperSBC, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Warmup to constant (uses optimizer's initial lr)
- **Batch size**: Polynomial curve from initial to final over total_steps
- **Weight decay**: Cosine annealing to zero (compensates for removed lr schedule)

### When to Use

Use when you want direct control over batch size scheduling with a polynomial curve. Useful for gradually increasing batch sizes during training to balance noise reduction and training speed.

---

## make_gnts_with_cosine_annealing_schedule

Creates an OptimizerWrapperGNTS with cosine annealing schedules.

### Signature

```python
def make_gnts_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    initial_threshold: float,
    final_threshold: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64
) -> Tuple[OptimizerWrapperGNTS, ScheduleAnything]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `initial_threshold` - Starting gradient norm threshold
- `final_threshold` - Ending gradient norm threshold
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)

### Returns

Tuple of (OptimizerWrapperGNTS, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Warmup to constant
- **Gradient norm threshold**: Inverse warmup to initial, then cosine anneal to final
- **Weight decay**: Warmup to full, then cosine anneal to zero

### When to Use

The flagship production algorithm. Use for adaptive batch sizing based on gradient quality. Directly controls gradient magnitude, eliminating need for learning rate scheduling.

---

## make_gnr_with_cosine_annealing_schedule_and_lr_to_constant

Creates an OptimizerWrapperGNR with gradient norm annealing and constant learning rate.

### Signature

```python
def make_gnr_with_cosine_annealing_schedule_and_lr_to_constant(
    optimizer: torch.optim.Optimizer,
    initial_norm: float,
    final_norm: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64
) -> Tuple[OptimizerWrapperGNR, ScheduleAnything]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `initial_norm` - Starting target gradient norm
- `final_norm` - Ending target gradient norm
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)

### Returns

Tuple of (OptimizerWrapperGNR, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Warmup to constant
- **Target gradient norm**: Cosine annealing from initial to final
- **Weight decay**: Warmup then cosine anneal to zero

### When to Use

Research control for isolating gradient direction vs magnitude effects. Rescales gradients to fixed norms, removing magnitude variability while keeping learning rate constant.

---

## make_gnr_with_cosine_annealing_schedule_and_lr_cosine_anneals

Creates an OptimizerWrapperGNR with both gradient norm and learning rate annealing.

### Signature

```python
def make_gnr_with_cosine_annealing_schedule_and_lr_cosine_anneals(
    optimizer: torch.optim.Optimizer,
    initial_norm: float,
    final_norm: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64
) -> Tuple[OptimizerWrapperGNR, ScheduleAnything]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `initial_norm` - Starting target gradient norm
- `final_norm` - Ending target gradient norm
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)

### Returns

Tuple of (OptimizerWrapperGNR, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Cosine annealing (follows same schedule as gradient norm)
- **Target gradient norm**: Cosine annealing from initial to final
- **Weight decay**: Warmup then cosine anneal to zero

### When to Use

Alternative GNR configuration where learning rate also anneals. Use when you want both gradient magnitude control and traditional learning rate decay.

---

## make_gns_with_cosine_annealing_schedule

Creates an OptimizerWrapperGNS with cosine annealing noise tolerance.

### Signature

```python
def make_gns_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    initial_tolerance: float,
    final_tolerance: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64
) -> Tuple[OptimizerWrapperGNS, ScheduleAnything]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `initial_tolerance` - Starting noise-to-signal tolerance
- `final_tolerance` - Ending noise-to-signal tolerance
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)

### Returns

Tuple of (OptimizerWrapperGNS, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Cosine annealing
- **Noise tolerance**: Cosine annealing from initial to final

### When to Use

Experimental gradient noise scale-based accumulation. Note: Did not perform well in practice. Included as research control.

---

## make_gns_default

Creates an OptimizerWrapperGNS with default schedule configuration.

### Signature

```python
def make_gns_default(
    optimizer: torch.optim.Optimizer,
    tolerance: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64
) -> Tuple[OptimizerWrapperGNS, ScheduleAnything]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `tolerance` - Noise-to-signal tolerance threshold
- `total_steps` - Total training steps
- `warmup_steps` - Steps for inverse warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)

### Returns

Tuple of (OptimizerWrapperGNS, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Cosine annealing
- **Noise tolerance**: Inverse warmup to tolerance, then constant

### When to Use

Default GNS configuration. Experimental - did not perform well in practice.

---

## make_mht_with_warmup_schedule

Creates an OptimizerWrapperMHT with warmup-to-constant statistical parameters.

### Signature

```python
def make_mht_with_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    confidence_level: float,
    percent_error_threshold: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64
) -> Tuple[OptimizerWrapperMHT, ScheduleAnything]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `confidence_level` - Target statistical confidence (e.g., 0.95 for 95%)
- `percent_error_threshold` - Maximum acceptable confidence interval width
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)

### Returns

Tuple of (OptimizerWrapperMHT, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Cosine annealing
- **Confidence level**: Warmup to constant (allows early rapid steps)
- **Percent error threshold**: Warmup to constant (allows early rapid steps)

### When to Use

Variance-based accumulation control. Steps when metric confidence interval is sufficiently narrow. Use when you have access to per-batch metrics (typically loss) and want low-variance updates.

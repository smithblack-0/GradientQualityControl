# Wrapper Factories API Guide

Convenience factory functions for creating pre-configured optimizer wrapper and schedule pairs. These factories encapsulate best-practice schedule configurations for common use cases. These are implemented using ScheduleAnything and returning a wrapped object, and a SynchronousSchedule which is a subclass of the torch learning rate object.

## Navigation

- **[User Guide](user_guide.md)** - Usage patterns and library overview
- **[Optimizer Wrapper API](optimizer_wrapper_api.md)** - Individual wrapper specifications
- **[Base Object API](base_object_api.md)** - Abstract base class for extending

---

## make_sbc_with_polynomial_schedule

Allows following a polynomial curve from initial batch size to final batch size with an included warmup; usable to schedule the batch size directly. Learning rate is warmed up to a constant, batch size warms up then follows a polynomial schedule, and weight decay warms up then executes cosine annealing to zero.

This is largely our interpretation of Smith's "Don't Decay the Learning Rate, Increase the Batch Size
". Do not expect an exact algorithm match, however, as we did not recheck the paper before implementing. Since the learning rate no longer decreases, but the regularization should, the weight decay is scheduled instead.

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
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None
) -> Tuple[OptimizerWrapperSBC, LRSchedule]:
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
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode. 


### Schedule Configuration

- **Learning rate**: Warmup to constant (uses optimizer's initial lr)
- **Batch size**: Polynomial curve from initial to final over total_steps, with direct warmup
- **Weight decay**: Direct warmup into cosine annealing to zero (compensates for removed lr schedule)
---

## make_sbc_with_polynomial_schedule_conventional_lr

Allows following a polynomial curve from initial batch size to final batch size with an included warmup; Unlike the last variant this continues to have a learning rate schedule and thus no weight decay scheduling. Learning rate warms up and then anneals to zero. No weight decay annealing. Behavior is otherwise like the original function.

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
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None
) -> Tuple[OptimizerWrapperSBC, LRSchedule]:
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
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode. 


### Schedule Configuration

- **Learning rate**: Warmup to constant (uses optimizer's initial lr) then decays to zero
- **Batch size**: Polynomial curve from initial to final over total_steps, with direct warmup
=
---

## make_gnts_with_cosine_annealing_schedule 

This implements a reactive system where the gradient quality is increased until the length of the gradient norm - which gets shorter as quality increases - is below a scheduled threshold. Then we step. Since this has the effect of directly regulating the length of the gradients, we omit the learning rate schedule.

Learning rate warms up to constant then just stays on, gradient norm threshold undergoes inverse warmup to starting value, then cosine anneals to ending value. Weight decay warms up to completely on, then cosine anneals to zero over timesteps, since the decay normally produced by learning rate is gone.

Creates an OptimizerWrapperGNTS with LRSchedules implementing these processes transparently for downstream objects.

### Signature

```python
def make_gnts_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    initial_threshold: float = 0.95,
    final_threshold: float = 0.25,
    warmup_multiplier: float = 10,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNTS, LRScheduler]:
```

### Parameters

- `optimizer` - Configured PyTorch optimizer. We will use it's learning rate and weight decay
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `initial_threshold` - Threshold annealing starts at once inverse warmup is complete. 
- `final_threshold` - Ending gradient norm threshold at end of training
- `warmup_multiplier` - This times initial threshold is where the inverse warmup starts at. Increase if the system does not step rapidly while in the warmup phase.
- `max_batch_draws` - Maximum accumulation (default: 64)
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode. 
- 
### Schedule Configuration

- **Learning rate**: Warmup to constant
- **Gradient norm threshold**: Inverse warmup to initial, then cosine anneal to final
- **Weight decay**: Warmup to full, then cosine anneal to zero


## make_gnts_with_cosine_annealing_schedule_conventional_lr 

This implements a reactive system where the gradient quality is increased until the length of the gradient norm - which gets shorter as quality increases - is below a scheduled threshold. Then we step. The cosine learning rate schedule is retained, eliminating any need for weight decay annealing. 

Creates an OptimizerWrapperGNTS with LRSchedules implementing these processes transparently for downstream objects.

### Signature

```python
def make_gnts_with_cosine_annealing_schedule_conventional_lr(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    initial_threshold: float = 0.95,
    final_threshold: float = 0.25,
    warmup_multiplier: float = 10,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNTS, LRScheduler]:
```

### Parameters

- `optimizer` - Configured PyTorch optimizer. We will use it's learning rate and weight decay
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `initial_threshold` - Threshold annealing starts at once inverse warmup is complete. 
- `final_threshold` - Ending gradient norm threshold at end of training
- `warmup_multiplier` - This times initial threshold is where the inverse warmup starts at. Increase if the system does not step rapidly while in the warmup phase.
- `max_batch_draws` - Maximum accumulation (default: 64)
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode.

### Schedule Configuration

- **Learning rate**: Warmup to constant then anneal down to zero.
- **Gradient norm threshold**: Inverse warmup to initial, then cosine anneal to final.

---

## make_gnr_with_cosine_annealing_schedule

The gradient norm rescaler class rescales the gradient norms to be a threshold size then immediately steps. The threshold is in turn scheduled, and this process directly controls the gradient length, omitting the need for a learning rate schedule. Since weight decay still needs to reduce, it is scheduled to cosine annealing instead decaying to zero. All properties undergo a normal warmup.

Creates an OptimizerWrapperGNR with gradient norm rescaling and constant learning rate.
### Signature

```python
def make_gnr_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    initial_norm: float,
    final_norm: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNR, LRSchedule]:
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `initial_norm` - Starting target gradient norm
- `final_norm` - Ending target gradient norm
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode.

### Schedule Configuration

- **Learning rate**: Warmup to constant.
- **Target gradient norm**: Cosine annealing from initial to final.
- **Weight decay**: Warmup then cosine anneal to zero.

---

## make_gns_with_cosine_annealing_schedule

Allows the scheduling of the gradient noise scale response quality with cosine annealing and thus adjusted the batch size.

The threshold is started high, using an inverse warmup, then comes down to the runtime value in inverse warmup. Then, it cosine anneals down to the final tolerance. This thus possesses an inverse response. A standard cosine annealing case, which anneals down to zero, is also attached. 

Creates an OptimizerWrapperGNS with cosine annealing noise tolerance.

### Signature

```python
def make_gns_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    initial_tolerance: float,
    final_tolerance: float,
    total_steps: int,
    warmup_steps: int,
    warmup_multiplier: int = 10,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNS, LRSchedule]:
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `initial_tolerance` - Starting noise-to-signal tolerance
- `final_tolerance` - Ending noise-to-signal tolerance
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode.

### Returns

Tuple of (OptimizerWrapperGNS, ScheduleAnything schedule)

### Schedule Configuration

- **Learning rate**: Warmup followed by Cosine annealing
- **Noise tolerance**: Inverse warmup followed by cosine annealing from initial to final


## make_gns_default

Creates an OptimizerWrapperGNS with default schedule configuration.

### Signature

```python
def make_gns_default(
    optimizer: torch.optim.Optimizer,
    tolerance: float,
    total_steps: int,
    warmup_steps: int,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNS, LRSchedule]:
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `tolerance` - Noise-to-signal tolerance threshold
- `total_steps` - Total training steps
- `warmup_steps` - Steps for inverse warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode.


### Schedule Configuration

- **Learning rate**: Warmup into Cosine annealing
- **Noise tolerance**: Inverse warmup to tolerance, then constant

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
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperMHT, LRSchedule]
```

### Parameters

- `optimizer` - Configured PyTorch optimizer
- `confidence_level` - Target statistical confidence (e.g., 0.95 for 95%)
- `percent_error_threshold` - Maximum acceptable confidence interval width
- `total_steps` - Total training steps
- `warmup_steps` - Steps for warmup phase
- `max_batch_draws` - Maximum accumulation (default: 64)
- `distributed_mode` - A specification which must not be none when distributed, but is optional otherwise. Tells us whether we are operating in a distributed data or sharded model distributed mode.

### Schedule Configuration

- **Learning rate**: Warmup into Cosine annealing
- **Confidence level**: Warmup to constant (allows early rapid steps)
- **Percent error threshold**: Warmup to constant (allows early rapid steps)

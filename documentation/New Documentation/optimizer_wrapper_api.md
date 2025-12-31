# Optimizer Wrapper API Reference

API reference for concrete optimizer wrapper implementations in Gradient Quality Control. For extending the base class, see [Base Object API](base_object_api.md). For usage patterns, see [User Guide](user_guide.md).

## Navigation

- **[User Guide](user_guide.md)** - Usage patterns and library overview
- **[Base Object API](base_object_api.md)** - Abstract base class specification
- **[Wrapper Factories API Guide](api_guide.md)** - Convenience factory functions

---

## OptimizerWrapperSBC

**Scheduled Batch Controller**

Research control for fixed accumulation schedules. Usable to ask for a multiple of a physical batch size to attempt to maintain a given logical batch size. 

### Constructor

The constructor wraps an optimizer, and asks for exactly as much additional information as is needed for the algorithm to run. Specifically, it asks for

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    physical_batch_size: int,
    max_batch_draws: int = 64
)
```

where

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `physical_batch_size` - Size of each microbatch
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)

### Schedule Targets

The following primary ScheduleAnything target is added

- **`logical_batch_size`** - Target total batch size injected by wrapper. Wrapper accumulates until reaching this size (rounded to nearest multiple of physical_batch_size).

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Algorithm

The system will keep track of num_draws and compute when 

```num_draws*physical_batch_size >= logical_batch_size.```

When this condition occurs, the step decision is taken. This ensures the effective batch size meets or exceeds the requested logical batch size, though it can only achieve sizes which are multiples of the physical batch size.

### Step

```python
def step(self) -> bool
```

**Returns:** True if optimizer stepped, False if still accumulating

---

## OptimizerWrapperGNTS

**Gradient Norm Threshold Scheduler**

Production algorithm for adaptive batch sizing based on gradient quality. Accumulates gradients until their magnitude falls below a configurable threshold, providing automatic batch size tuning and gradient noise reduction.

### Constructor

The constructor wraps an optimizer, and asks for exactly as much additional information as is needed for the algorithm to run. Specifically, it asks for

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    max_batch_draws: int = 64
)
```

where

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)

### Schedule Targets

The following primary ScheduleAnything target is added

- **`gradient_norm_threshold`** - Target gradient norm threshold injected by wrapper. Wrapper accumulates until gradient norm falls at or below this threshold.

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Algorithm

On each step call, the system computes the current gradient norm (divided by num_draws to get the mean) and checks whether

```mean_gradient_norm <= gradient_norm_threshold```

When this condition is satisfied for all parameter groups, the step decision is taken. As accumulation continues, the mean gradient norm typically decreases, making it progressively more likely to meet the threshold. The system will force a step when `num_draws >= max_batch_draws` regardless of gradient norm.

### Step

```python
def step(self) -> bool
```

**Returns:** True if optimizer stepped, False if still accumulating

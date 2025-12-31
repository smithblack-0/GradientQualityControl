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

---

## OptimizerWrapperGNR

**Gradient Norm Rescaler**

Research control for isolating gradient direction effects from magnitude effects. Rescales all gradients to a fixed target norm before every optimizer step, ensuring constant gradient magnitude across training.

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

- **`target_gradient_norm`** - Target gradient norm injected by wrapper. Wrapper rescales all gradients to match this norm before stepping.

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Algorithm

On each step call, the system computes the current gradient norm globally across all parameters, then rescales all gradients by the factor

```rescale_factor = target_gradient_norm / current_gradient_norm```

This ensures the gradient direction is preserved while the magnitude is set to exactly `target_gradient_norm`. The optimizer is then stepped with the rescaled gradients. This wrapper always steps on every call, never accumulating.

### Step

```python
def step(self) -> bool
```

**Returns:** True (always steps)

---

## OptimizerWrapperMHT

**Metric Hypothesis Test**

Statistical wrapper for variance-based accumulation control. Accumulates metric samples (typically loss) across batches and uses a two-tailed t-test to determine when the confidence interval is sufficiently narrow, ensuring low-variance parameter updates.

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

The following primary ScheduleAnything targets are added

- **`confidence_level`** - Statistical confidence level for t-test (e.g., 0.95 for 95% confidence) injected by wrapper
- **`percent_error_threshold`** - Maximum acceptable confidence interval width injected by wrapper

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Algorithm

On each step call, the user provides a metric value (typically loss). The system accumulates these metric samples and performs a two-tailed t-test to compute the confidence interval at the specified `confidence_level`. The step decision is taken when the confidence interval meets the `percent_error_threshold` criterion, indicating the metric estimate has sufficiently low variance. The system will force a step when `num_draws >= max_batch_draws` regardless of confidence interval width.

### Step

```python
def step(self, metric: float) -> bool
```

**Parameters:**
- `metric` - Metric value for this batch (typically loss)

**Returns:** True if optimizer stepped, False if still accumulating

---

## OptimizerWrapperGNS

**Gradient Noise Scale**

Research control inspired by McCandlish et al.'s gradient noise scale theory for adaptive batch sizing.

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

- **`gradient_noise_to_signal_tolerance`** - Noise-to-signal ratio threshold injected by wrapper

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Algorithm

On each step call, the system tracks per-microbatch gradient norms and estimates the gradient noise scale using:

```
estimated_GNS = Var(||g_i||) / E[||g_i||²]
```

where `||g_i||` is the gradient norm for each accumulated microbatch. The step decision is taken when

```estimated_GNS <= num_draws * gradient_noise_to_signal_tolerance```

This criterion, inspired by McCandlish et al.'s work, balances noise reduction benefits against accumulation costs. The system will force a step when `num_draws >= max_batch_draws` regardless of the GNS estimate.

### Step

```python
def step(self) -> bool
```

**Returns:** True if optimizer stepped, False if still accumulating

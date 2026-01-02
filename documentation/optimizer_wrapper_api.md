# Optimizer Wrapper API Reference

API reference for concrete optimizer wrapper implementations in Gradient Quality Control. For extending the base class, see [Base Object API](base_object_api.md). For usage patterns, see [User Guide](user_guide.md).

## Navigation

- **[User Guide](user_guide.md)** - Usage patterns and library overview
- **[Base Object API](base_object_api.md)** - Abstract base class specification
- **[Wrapper Factories API Guide](api_guide.md)** - Convenience factory functions

---

## OptimizerWrapperSBC

 A *control* controller for scheduling the logical batch size. It exposes for scheduling the entries "lr", "weight_decay", and "logical_batch_size". It is initialized with physical batch size, and will only invoke step once logical batch size from all steps to now exceed physical batch size. Generally, proper usage involves warming up the learning rate to a constant, warming up then cosine annealing the weight decay, and using some sort of inverse schedule, such as a low-to-high polynomial schedule, to start asking for a small number of batches that progressively increase.

### Constructor

The constructor wraps an optimizer, and asks for exactly as much additional information as is needed for the algorithm to run. Specifically, it asks for

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    physical_batch_size: int,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None
)
```

where

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `physical_batch_size` - Size of each microbatch
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)
- `distributed_mode` - One of "replicated", "sharded", influencing what a batch is considered to be.
### Schedule Targets

The following primary ScheduleAnything target is added

- **`logical_batch_size`** - Target total batch size injected by wrapper. Wrapper accumulates until reaching this size (rounded to nearest multiple of physical_batch_size).

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Distributed Support

We presume the same batch size is used on all devices. When it is detected that this is operating in a distributed environment, the distributed flag must be set to one of "replicated" or "sharded". These have the following effects:

- **Replicated**: Each batch on each device counts individually. physical batch size is multiplied by the number of devices
- **Sharded**: The batches are the same on all devices. No change.

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

The current best production algorithm. Accumulates gradients until their magnitude falls below a configurable threshold, providing automatic batch size tuning and gradient noise reduction. This is known to produce certain quality guarantees. See research for those details.

The features "lr", "weight_decay", and "gradient_norm_threshold" are primary schedule targets. Normal operation is to warmup learning rate to a constant, warmup weight decay then cosine anneal, and inverse warmup then cosine anneal the gradient norm threshold. Like the Gradient Norm Rescaler this directly controls the length of the gradients, eliminating the need for a learning rate schedule and thus producing more consistent Adam updates.

### Constructor

The constructor wraps an optimizer, and asks for exactly as much additional information as is needed for the algorithm to run. Specifically, it asks for

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None
)
```

where

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)
- `distributed_mode` - One of "replicated", "sharded". Replicated is used for data parallel processes like DDP, while sharded for model parallel processes. These influence how to merge metrics.

### Schedule Targets

The following primary ScheduleAnything target is added

- **`gradient_norm_threshold`** - Target gradient norm threshold injected by wrapper. Wrapper accumulates until gradient norm falls at or below this threshold.

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Distributed Support

When it is detected that a distributed mode is being used, the distributed mode flag must be set to either replicated or sharded. These have the following behaviors

- **`replicated`**: No change in behavior. Gradient norms are replicated before the system gets a chance to check, and we are automatically taking the norms of synchronized gradients, so no change is needed.
- **`sharded`**: The norm on each device is presumed to be part of the whole norm, and needs to be added up. We use the decomposition sqrt(sum(grad_norm^2)) to equivalently add up the norms from each device to get the same norm on all devices. 

Cases involving both replication and sharding are not currently supported. Submit a pull request if interested. Under the hood we use torch distributed utilities. 

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

A *control* wrapper which always steps, but also rescales the gradients to a constant scale during all steps before doing so. It exposes "weight_decay", "gradient_norm_target", and "lr" for scheduling. When used, it looks up the current gradient norm target, figures out the gradient norm, then rescales all gradients to that norm. It then immediately steps and zeros the gradients in the optimizer.

Usage is estimated to be best as learning rate to constant warmup, weight decay to warmup then cosine annealing, and cosine annealing of norm. This is because it replaces learning rate scheduling by directly controlling the length of the gradient instead.

Used to isolate how much gain comes from more consistent gradient lengths, which has synergetic effects with Adam optimizers and optimizers with second-moment curvature estimation.

### Constructor

The constructor wraps an optimizer, and asks for exactly as much additional information as is needed for the algorithm to run. Specifically, it asks for

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
    mode: Literal["global", "independent"] = "global"
)
```

where

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)
- `distributed_mode` - One of "replicated", "sharded". Replicated is used for data parallel processes like DDP, while sharded for model parallel processes. These influence how to merge metrics.
- `mode` - Scaling mode: "global" computes norm across all parameters and scales uniformly, "independent" scales each parameter to target norm separately (default: "global")

### Schedule Targets

The following primary ScheduleAnything target is added

- **`target_gradient_norm`** - Target gradient norm injected by wrapper. Wrapper rescales all gradients to match this norm before stepping.

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Distributed Support

When it is detected that a distributed mode is being used, the distributed mode flag must be set to either replicated or sharded. These have the following behaviors

- **`replicated`**: No change in behavior. Gradient norms are replicated during backwards pass automatically so no change is needed.
- **`sharded`**: The norm on each device is presumed to be part of the whole norm, and needs to be added up. We use the decomposition sqrt(sum(grad_norm^2)) to equivalently add up the norms from each device to get the same norm on all devices. 

Cases involving both replication and sharding are not currently supported. Submit a pull request if interested.

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

A controller that makes it's step decision based on variance in an observed metric. Once, given the metric samples, the controller is %confidence_level sure that the true metric has been found such that the confidence interval is under %percent_error. Features "lr", "confidence_level", and "percent_error_threshold" are exposed to ScheduleAnything. 

 Accumulates metric samples (typically loss) across batches/devices and uses a two-tailed t-test to determine when the confidence interval is sufficiently narrow, ensuring low-variance parameter updates. We step, in other words, when we are confident what the true metric is. A typical use case would have one scheduling the learning rate using a cosine schedule, and confidence level and percent error threshold both warmup to a constant to allow early steps to proceed rapidly.

### Constructor

The constructor wraps an optimizer, and asks for exactly as much additional information as is needed for the algorithm to run. Specifically, it asks for

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
)
```

where

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)
- `distributed_mode` - One of "replicated", "sharded". Replicated is used for data parallel processes like DDP, while sharded for model parallel processes. These influence how to merge metrics.

### Schedule Targets

The following primary ScheduleAnything targets are added

- **`confidence_level`** - Statistical confidence level for t-test (e.g., 0.95 for 95% confidence) injected by wrapper
- **`percent_error_threshold`** - Maximum acceptable confidence interval width injected by wrapper

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Distributed Support

The primary issue is whether or not samples are independent.

- **`replicated`**: We presume the independence of samples. All metric draws from all devices are appended to the list.
- **`sharded`**: This is still just one batch. We average the metric to be safe, and just consider it one sample.

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

Research control inspired by McCandlish et al.'s gradient noise scale theory for adaptive batch sizing. Largely a failed mechanism that did not correctly optimize performance, as it did not perform well under Adam.

It exposes "lr" and "noise_tolerance"; noise tolerance is literally what the noise to signal ratio we are willing to tolerate. It delays taking steps until the GNS is below a threshold weighted by the cost of processing one more batch. 

### Constructor

The constructor wraps an optimizer, and asks for exactly as much additional information as is needed for the algorithm to run. Specifically, it asks for

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
)
```

where

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)
- `distributed_mode` - One of "replicated", "sharded". Replicated is used for data parallel processes like DDP, while sharded for model parallel processes. These influence how to merge metrics.

### Schedule Targets

The following primary ScheduleAnything target is added

- **`noise_tolerance`** - Noise-to-signal ratio threshold injected by wrapper

In addition the following two are almost always present on Adam optimizer derivatives

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)

### Distributed Support

The primary issue is whether or not samples are independent. 

- **`replicated`**: We presume the independence of samples. All metric draws from all devices are appended to the list on all devices. 
- **`sharded`**: This is still just one batch. We use the decomposition sqrt(sum(grad_norm^2)) to equivalently add up the norms from each device to get the same norm on all devices. 

### Algorithm

On each step call, the system tracks per-microbatch gradient norms and estimates the gradient noise scale using:

```
estimated_GNS = Var(||g_i||) / E[||g_i||²]
```

where `||g_i||` is the gradient norm for each accumulated microbatch. The step decision is taken when

```estimated_GNS <= num_draws * noise_tolerance```

This criterion, inspired by McCandlish et al.'s work, balances noise reduction benefits against accumulation costs. The system will force a step when `num_draws >= max_batch_draws` regardless of the GNS estimate.

### Step

```python
def step(self) -> bool
```

**Returns:** True if optimizer stepped, False if still accumulating

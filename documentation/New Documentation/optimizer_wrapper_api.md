# Optimizer Wrapper API Reference

Complete API specification for all optimizer wrappers in Gradient Quality Control.

This document defines the exact interface contracts that all code must conform to. Think of this as a contract that could be handed to an implementation fairy with pixie dust - if she follows this spec exactly, it will work correctly.

## Navigation

- **[User Guide](user_guide.md)** - Usage patterns and concepts
- **[Wrapper Factories API Guide](api_guide.md)** - Convenience factory functions
- **[Research Guide](research_guide.md)** - Research background and theory
- **[README](../README.md)** - Installation and quick start

---

## AbstractOptimizerWrapper

The base class for all GQC optimizer wrappers implementing the Sequential Binary Decision Controller (SBDC) pattern. 

### Purpose

Implementing a Sequential Binary Decision Controller for gradient accumulation requires solving several tedious problems. The gradients must be accumulated correctly across batches and averaged before stepping. The wrapper must track how many batches have been processed, how many optimizer steps have been taken, and how many batches are currently accumulated. This state must serialize and restore correctly for checkpointing. The wrapper must behave transparently like a normal optimizer so existing code continues to work. Statistics about the control algorithm's behavior must be collected and reported. These problems are identical across all control algorithms - only the decision logic varies.

Despite this, ultimately, the differences between algorithms are subtle. All algorithms ultimately make one decision:

1) Do we step the optimizer then zero the grads?
2) Do we fiddle with the gradients at all before taking that step?

This leaves us with a strong candidate for abstraction. It is this niche the base class exists to solve. I

* **Manages Gradient Accumulation** by dividing accumulated gradients by the number of batches when stepping the optimizer, converting PyTorch's natural summation into the required mean.
* **Tracks Statistics**  through `_received_batch()` and `_take_optimizer_step()`, maintaining invariants like ensuring at least one batch is drawn before any step and preventing accumulation beyond `max_draws`.
* **Accelerates Serialization** through `state_dict()` and `load_state_dict()`, automatically saving and restoring all wrapper state alongside the wrapped optimizer's state. Subclasses store their algorithm parameters through `set_state()`, which writes to an internal dictionary that serializes automatically - direct field assignment after initialization throws an error to prevent serialization bugs.
* **Handles Statistical Reporting**  through `statistics()` and `vital_statistics()`, pulling from both wrapper state and optimizer parameters and aggregating across parameter groups when values differ.
* **Integrates Cleanly** y subclassing `torch.optim.Optimizer` and delegating all non-overridden methods and attributes to the wrapped optimizer. `.step()` then may or may not step the wrapped optimizer, and 

### Public Instance Behavior

Instances of this object pretend to be, almost completely, an optimizer instance. This means the object is implemented as on optimizer subclass, and downstream code can invoke the .step field as normal, at which point the underlying subclasses's step algorithm is used to judge whether or not to take a step.  

The following fields are available, with indicated behavior, on the main class

**Public Fields**:
 **`optimizer`** (`torch.optim.Optimizer`) - The wrapped optimizer, directly accessible
- **`wrapper_states`** (`Dict`) - Internal state storage. **Direct access is undefined behavior.** Use `get_state()` and `set_state()` instead.

**Public Methods**
- **`step`**: Transparently duck-types exactly as before. 
- **`zero_grad`**: Now throws. The wrapper resets grads instead.
- **`statistics`**: Returns complete dictionary of features governing internal behavior.
- **`vital_statistics`**: Returns dictionary of things to display or log that are vital performance indicators; subclasses must judge and mark whether somethign is vital.
- **`state_dict`**: Gets the state dict, storing the wrapped optimizer and the state dict from the layer.
- **`load_state_dict`**: Losslessly resumes from a state dict.

All other methods or calls are automatically and transparently passed into the wrapped optimizer. If you call, for instance, the param_groups field you get the group off the base optimizer.

### Subclassing Instance Behavior

Using the subclass to implement an algorithm requires knowledge of a few contract details. First, and perhaps most important, is what a subclass should usually implement. Under standard conditions, you are responsible for implementing only one thing

* **`step`**: Including deciding when to take optimizer steps, when to just advance without taking steps.

However, in the process of doing this, there are some rules that are needed to allow the automatic mechanisms in the base class to work. Specifically you should:

* Invoke `_received_batch` when first entering step to update the batch counter system for statistics.
* Invoke `_take_optimizer_step` any time it is decided an optimizer step is needed.
* Initialize, store and update state through the "set_state" and "get_state" method. 

Attempts to initialize and set fields on the class directly will throw, and in fact must go through set_state instead. set_state lets you set a statistic as 'vital', and one should do so with important fields like, for example, num_last_batch_draws or last_mean_gradient_norm. get_state can get optimizer state as well, and when doing so will return a list rather than a single entry from the optimizer parameter groups; a special flag can ask for the mean, max, or min instead. 


---


The wrapper transparently forwards all optimizer methods and attributes you don't override. When users call `wrapper.param_groups` or `wrapper.add_param_group()`, it works automatically - you don't write delegation code. The wrapper satisfies `isinstance(wrapper, Optimizer)` and works with ScheduleAnything and other PyTorch tooling without extra effort.

Finally, it tracks statistics for you. Counters like `num_batches`, `num_steps`, and `num_draws` update automatically when you call `_received_batch()` and `_take_optimizer_step()`. The gradient norm from the last step is computed and cached. The `statistics()` and `vital_statistics()` methods pull from your state automatically - you mark what's vital via the `flag` parameter in `set_state()`, and the base class handles filtering and aggregation across parameter groups.

Without the base class, every algorithm would reimplement this infrastructure. With it, you write your control logic and call three methods: `_received_batch()`, `_take_optimizer_step()`, and `set_state()`. The rest just works.

---


## Underlying Details

*This section explains how the base class achieves the contract. This helps implementers understand the mechanism, but is not part of the contract itself.*

### Gradient Accumulation Mechanism

PyTorch's `.backward()` accumulates gradients by default - it sums them into `.grad`. After N consecutive backward passes without zeroing, each parameter's `.grad` contains the sum of N batches' gradients.

The wrapper exploits this: when `_take_optimizer_step()` is called, it multiplies all gradients by `1/N` (where N = `num_draws`) to convert the sum into a mean. The optimizer then steps with these averaged gradients.

This is how the wrapper adaptively controls effective batch size in constant memory - accumulation happens naturally, averaging happens on-demand.

### State Management Design

All wrapper state lives in `wrapper_states`, a dictionary storing: `{name: {"value": X, "flag": "vital"/"optional"}}`.

The only interface is `set_state()` and `get_state()`. Why? **Serialization guarantee.** When `state_dict()` is called, it simply dumps `wrapper_states` + the wrapped optimizer's state. When `load_state_dict()` restores, it reloads this dict.

If subclasses set fields directly (e.g., `self.my_threshold = 0.5`), those fields won't be in `wrapper_states` and won't serialize. To prevent this bug, direct field setting after initialization throws an error, forcing use of `set_state()`.

### Transparent Optimizer Proxying

The wrapper implements `__getattribute__` and `__setattr__` to intercept all attribute access:
- If the attribute exists in the wrapper class hierarchy (e.g., `step`, `statistics`): use wrapper's version
- Otherwise: forward to the wrapped optimizer

This is why `wrapper.param_groups`, `wrapper.state`, and `wrapper.add_param_group()` all work transparently - they're automatically forwarded to the underlying optimizer. The wrapper satisfies `isinstance(wrapper, Optimizer)` and works with existing PyTorch tooling.

### Counter Tracking

The built-in counters (`num_batches`, `num_steps`, `num_draws`) are stored via `set_state()` and marked vital:
- `_received_batch()` increments `num_batches` and `num_draws`
- `_take_optimizer_step()` increments `num_steps` and resets `num_draws` to 0

Because they're in `wrapper_states`, they automatically serialize/deserialize correctly.

### Statistics Filtering

Both `statistics()` and `vital_statistics()` pull from two sources:
1. **wrapper_states**: All entries (statistics) or only vital-flagged (vital_statistics)  
2. **optimizer.param_groups**: All float-valued keys in both cases

For multi-group parameters:
- If all groups have same value: return as-is (e.g., `"lr": 0.001`)
- If groups differ: add `*` suffix and aggregate (e.g., `"lr*": 0.0015` with mean)

The `aggregate_lists` parameter in `get_state()` controls aggregation: `None` (return list), `"mean"`, `"max"`, or `"min"`.

---


## Implementation Details

The base class implements the following responsibilities

1) Provide support for drawing multiple batches and average gradients when it is time to take an optimizer step.
2) 
All information is stored in one of two places. These places are

* 

### Constructor

```python
def __init__(self, optimizer: torch.optim.Optimizer, max_draws: int)
```

**Parameters:**
- **`optimizer`** - The PyTorch optimizer to wrap
- **`max_draws`** - Maximum number of batches that can accumulate before forcing an optimizer step

**Initializes:**
- Wraps the optimizer
- Sets up wrapper state: `num_batches`, `num_steps`, `num_draws`
- Configures `max_draws`

---

### step (Abstract)

```python
def step(self, closure: Optional[Callable[[], Any]] = None) -> bool
```

**Abstract - must be implemented by subclasses.**

**Parameters:**
- **`closure`** - Optional closure (e.g., for LBFGS). **Support required by contract.**

**Returns:**
- **`bool`** - `True` if optimizer stepped this call, `False` if still accumulating

**Contract:**
- Must return bool indicating whether optimizer stepped
- Must call `_batch_received()` to update counters
- Should call `_take_optimizer_step(closure)` when deciding to step
- Call once per batch in training loop

---

### statistics

```python
def statistics(self) -> Dict[str, Any]
```

**Returns complete statistics dictionary.**

**Contents:**
- All entries in `wrapper_states` (vital and optional)
- All float-valued keys from `optimizer.param_groups`
- For multi-group params with different values: key + `*` suffix, mean value
- For multi-group params with same value: key without suffix

**Properties:**
- Read-only (never modifies state)
- Deterministic (same state → same output)
- Can call multiple times per step
- Can call before first step

**Example:**
```python
{
    "num_batches": 150,
    "num_steps": 25,
    "num_draws": 2,
    "last_grad_norm": 0.342,
    "gradient_norm_threshold": 0.5,  # from wrapper_states
    "lr": 0.001,  # same across groups
    "weight_decay*": 0.0055  # mean of different values
}
```

---

### vital_statistics

```python
def vital_statistics(self) -> Dict[str, Any]
```

**Returns curated statistics for tqdm/logging.**

**Contents:**
- All wrapper_states entries marked `flag="vital"`
- `num_batches` and `num_draws` (always vital)
- All float-valued keys from `optimizer.param_groups`
- Same aggregation rules as `statistics()` (mean with `*` for different values)

**Properties:**
- Read-only
- Deterministic
- Subset of `statistics()`
- Can call before first step

**Purpose:**
The "health dashboard" - key metrics for progress bars and training dashboards.

---

### state_dict

```python
def state_dict(self) -> Dict[str, Any]
```

**Contract:**
If you `state = wrapper.state_dict()`, restart process, create new wrapper, then `wrapper.load_state_dict(state)`, training resumes **exactly** where it left off. No observable difference from never stopping.

**Preserves:**
- All `wrapper_states` (vital and optional)
- Complete `optimizer.state_dict()`
- All counters and cached values

---

### load_state_dict

```python
def load_state_dict(self, state_dict: Dict[str, Any]) -> None
```

**Parameters:**
- **`state_dict`** - State from `state_dict()`

**Contract:**
Restores wrapper to exact state when `state_dict()` was called. Training continues seamlessly.

---

### zero_grad

```python
def zero_grad(self) -> None
```

**Raises:** `NotImplementedError` - Always.

**Contract:**
Users must **never** call `zero_grad()` with optimizer wrappers. The wrapper manages gradient clearing. This method exists only to fail fast with a clear error instead of silent corruption.

**Rationale:**
Wrappers accumulate gradients across batches. Calling `zero_grad()` breaks accumulation.

---

### set_state (For Subclasses)

```python
def set_state(self, name: str, value: Any, flag: Literal["vital", "optional"]) -> None
```

**For subclass implementation - stores wrapper state.**

**Parameters:**
- **`name`** - State variable name
- **`value`** - Value to store (must be serializable)
- **`flag`** - `"vital"` (appears in `vital_statistics()`) or `"optional"` (only in `statistics()`)

**Contract:**
- Stores in `wrapper_states`
- `flag` parameter required (no default)
- If name exists: updates value, flag must match (throws if flag changes)
- If name exists in `optimizer.param_groups`: **throws** (namespace collision)
- Creates new entry on first call

**Throws:**
- If `name` matches optimizer param_group key (e.g., "lr", "weight_decay")
- If `name` exists but `flag` differs from stored flag

---

### get_state (Unified Access)

```python
def get_state(self, name: str) -> Any
```

**Unified interface for both wrapper and optimizer state.**

**Returns:** The state value (just the value, not metadata)

**Contract - Search Order:**
1. If `name` in `wrapper_states`: return that value
2. Else if `name` in `optimizer.param_groups`:
   - All groups same value → return that value
   - Groups have different values → return mean
3. Else: crash (not found)

**Purpose:**
Transparent access to both wrapper state (`num_batches`, custom thresholds) and optimizer state (`lr`, `weight_decay`) through one interface.

**Example:**
```python
threshold = wrapper.get_state("gradient_norm_threshold")  # from wrapper_states
lr = wrapper.get_state("lr")  # from optimizer.param_groups
batches = wrapper.get_state("num_batches")  # from wrapper_states (vital)
```

---

### _batch_received (Protected)

```python
def _batch_received(self) -> None
```

**For subclasses - required to use.**

**Contract:**
- Increments `num_batches` and `num_draws` via `set_state()`
- Throws if `num_draws` would exceed `max_draws`
- Subclasses **must** call once per batch (typically at start of `step()`)

**Purpose:**
Centralized counter management to maintain invariants.

---

### _take_optimizer_step (Protected)

```python
def _take_optimizer_step(self, closure: Optional[Callable[[], Any]] = None) -> None
```

**For subclasses - required to use when stepping.**

**Contract - Execution Order:**
1. Average gradients: multiply all by `1 / num_draws`
2. Compute L2 gradient norm across all parameters
3. Step wrapped optimizer with averaged gradients
4. Zero all gradients
5. Update state: store gradient norm, increment `num_steps`, reset `num_draws` to 0
6. Cache optimizer return value (if closure provided)

**Parameters:**
- **`closure`** - Passed to `optimizer.step(closure)`

**Throws:**
- If `num_draws == 0` (cannot step without accumulated batches)

**Purpose:**
Ensures gradient averaging invariant maintained consistently. Subclasses **must** use this to step - bypassing it violates the contract.

---

### Invariants

Properties that **must always hold:**

1. **`num_batches >= num_steps`** - Cannot step more than batches processed
2. **`num_draws <= max_draws`** - Cannot exceed maximum accumulation
3. **Gradient averaging** - Gradients always averaged before stepping
4. **One batch minimum** - At least one batch before any step
5. **State preservation** - `state_dict()` → `load_state_dict()` preserves exact state
6. **Transparent forwarding** - Non-overridden methods/attributes forward to `optimizer`
7. **Required utilities** - Subclasses must use `_batch_received()` and `_take_optimizer_step()`
8. **Closure support** - All subclasses must support closure parameter
9. **Namespace separation** - `set_state()` cannot use optimizer param_group names
10. **Immutable flags** - Vital/optional status cannot change once set
11. **Anytime statistics** - `statistics()` and `vital_statistics()` work before first step

---

### Mathematical Model

Let $b_i$ = gradients from batch $i$, $n$ = `num_draws`, $\theta$ = model parameters.

**Accumulation:**
$$g_{accumulated} = \sum_{i=1}^{n} b_i$$

**Averaging (in `_take_optimizer_step()`):**
$$g_{mean} = \frac{1}{n} \sum_{i=1}^{n} b_i$$

**Gradient Norm (L2):**
$$\|g_{mean}\|_2 = \sqrt{\sum_{p \in \theta} \|g_{mean}[p]\|^2}$$

Stored as `last_grad_norm`, accessible via `get_state("last_grad_norm")`.

---

### Subclass Implementation Contract

**Subclasses must:**

1. **Implement `step(closure=None) -> bool`**
   - Return True/False for stepped/accumulating
   - Call `_batch_received()` once per batch
   - Call `_take_optimizer_step(closure)` when stepping
   - Support closure

2. **Use state management**
   - Store all state via `set_state(name, value, flag)`
   - Retrieve via `get_state(name)`
   - Never directly access `wrapper_states`

3. **Call `super().__init__(optimizer, max_draws)`** before subclass setup

**Subclasses may:**
- Override `statistics()` and `vital_statistics()` (call `super()` and extend)
- Implement any control algorithm (fixed accumulation, quality thresholds, rescaling, etc.)

---

## Under the Hood

*Implementation notes - not part of contract:*

**Transparent forwarding:** `__getattribute__` and `__setattr__` intercept attribute access, forwarding to wrapped optimizer for anything not in wrapper class hierarchy.

**wrapper_states storage:** Dictionary where `set_state()` writes `{name: {"value": X, "flag": "vital"/"optional"}}`. Methods filter by flag when building statistics.

**Gradient averaging:** PyTorch's `.backward()` accumulates (sums) gradients. After $n$ backward passes, `_take_optimizer_step()` multiplies by $1/n$ to convert sum → mean.

---

## Concrete Implementations

*To be documented:*

- **OptimizerWrapperGNTS** - Gradient Norm Threshold Scheduler (flagship)
- **OptimizerWrapperSBC** - Scheduled Batch Controller
- **OptimizerWrapperGNR** - Gradient Norm Rescaler
- **OptimizerWrapperGNS** - Gradient Noise Scale
- **OptimizerWrapperMHT** - Metric Hypothesis Test

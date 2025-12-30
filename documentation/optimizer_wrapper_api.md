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

On each batch, the wrapper decides: step the optimizer, or continue accumulating gradients? This is the core Sequential Binary Decision - the foundation of adaptive gradient quality control.

---

### Objects (Fields)

**Public:**
- **`optimizer`** (`torch.optim.Optimizer`) - The wrapped optimizer, directly accessible
- **`wrapper_states`** (`Dict`) - Internal state storage. **Direct access is undefined behavior.** Use `get_state()` and `set_state()` instead.

**Contract on attribute access:**
The wrapper behaves transparently like the wrapped optimizer for all attributes not explicitly overridden. Accessing `wrapper.param_groups`, `wrapper.state`, `wrapper.defaults`, etc. forwards to the wrapped optimizer.

---

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

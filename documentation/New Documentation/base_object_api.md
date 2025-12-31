# Optimizer Wrapper Base clase

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

The base class solves these problems once. It manages gradient accumulation by dividing accumulated gradients by the number of batches when stepping the optimizer, converting PyTorch's natural summation into the required mean. It tracks counters through `_received_batch()` and `_take_optimizer_step()`, maintaining invariants like ensuring at least one batch is drawn before any step and preventing accumulation beyond `max_draws`. It provides serialization through `state_dict()` and `load_state_dict()`, automatically saving and restoring all wrapper state alongside the wrapped optimizer's state. Subclasses store their algorithm parameters through `set_state()`, which writes to an internal dictionary that serializes automatically - direct field assignment after initialization throws an error to prevent serialization bugs. The base class handles statistics reporting through `statistics()` and `vital_statistics()`, pulling from both wrapper state and optimizer parameters and aggregating across parameter groups when values differ. It implements transparent optimizer interface forwarding by subclassing `torch.optim.Optimizer` and delegating all non-overridden methods and attributes to the wrapped optimizer.

The base class does not implement the control algorithm. Subclasses implement `step()` to decide when conditions warrant calling `_take_optimizer_step()`. The base class provides infrastructure; subclasses provide control logic. Overall, the user experience should be that there is a fairly normal optimizer that just calls .zero_grad itself.

### ScheduleAnything Integration

Optimizer wrappers integrate seamlessly with ScheduleAnything to enable dynamic scheduling of both optimizer parameters (lr, weight_decay) and wrapper-specific control parameters (gradient_norm_threshold, logical_batch_size, etc.). This integration happens through parameter extension and unified access.

**How it works:**

When a subclass calls `set_state(name, value, "optimizer")`, the base class uses ScheduleAnything to extend the wrapped optimizer's param_groups with that new parameter. The parameter gets injected into every parameter group alongside native optimizer parameters like lr. After extension, ScheduleAnything schedules can bind to and modify this parameter just like they would lr or weight_decay.

For example, OptimizerWrapperGNTS needs to schedule `gradient_norm_threshold`. During `__init__`, it calls `set_state("gradient_norm_threshold", 1.0, "optimizer")`. This extends the AdamW optimizer to include gradient_norm_threshold in its param_groups. A ScheduleAnything schedule can then bind to both lr and gradient_norm_threshold, warming up lr while annealing the threshold. 

The wrapper exposes `.valid_schedule_targets` to list all schedulable parameters. This includes the optimizer's native parameters (discovered by inspecting the first param_group) plus any wrapper-specific parameters added via the "optimizer" flag. Factory functions use this to verify schedule bindings are valid.

**Key mechanisms:**

- **Parameter extension:** `set_state(name, value, "optimizer")` → ScheduleAnything injects parameter into all param_groups
- **Discovery:** `.valid_schedule_targets` → lists all schedulable parameters (native + extended)
- **Unified access:** `get_state(name)` → retrieves from wrapper_states or optimizer param_groups transparently
- **Standard patterns:** Factory functions (make_gnts_with_cosine_annealing_schedule) create wrapper and schedule together, binding to multiple parameters

This design means control algorithms can expose their parameters for scheduling without requiring special schedule implementations. Schedules work with wrapper parameters identically to optimizer parameters.

### Public Instance Behavior

Instances of this object pretend to be, almost completely, an optimizer instance. This means the object is implemented as on optimizer subclass, and downstream code can invoke the .step field as normal, at which point the underlying subclasses's step algorithm is used to judge whether or not to take a step.  

The following fields are available, with indicated behavior, on the main class

**Public Fields**:
- **`optimizer`** (`torch.optim.Optimizer`) - The wrapped optimizer, directly accessible
- **`wrapper_states`** (`Dict`) - Internal state storage. **Direct access is undefined behavior.** Use `get_state()` and `set_state()` instead.

**Public Properties**:
- **`valid_schedule_targets`** (`List[str]`) - Read-only list of all schedulable parameter names. Includes native optimizer parameters (lr, weight_decay, momentum, etc.).

**Public Methods**
- **`step`**: Transparently duck-types exactly as before. 
- **`zero_grad`**: Now throws. The wrapper resets grads instead.
- **`statistics`**: Returns complete dictionary of features governing internal behavior.
- **`vital_statistics`**: Returns dictionary of things to display or log that are vital performance indicators; subclasses must judge and mark whether somethign is vital.
- **`state_dict`**: Gets the state dict, storing the wrapped optimizer and the state dict from the layer.
- **`load_state_dict`**: Losslessly resumes from a state dict.

All other methods or calls are automatically and transparently passed into the wrapped optimizer. If you call, for instance, the param_groups field you get the group off the base optimizer.

### Subclassing Instance Behavior

Using the subclass to implement an algorithm requires knowledge of a few contract details. First, and perhaps most important, is what a subclass should usually implement. Under standard conditions, you are responsible for implementing only one thing: step.

**Subclasses must:**

1. **Implement `step() -> bool`**
   - Return True/False for stepped/accumulating
   - Call `_batch_received()` once per batch
   - Call `_take_optimizer_step()` when stepping

2. **Use state management**
   - Store all state via `set_state(name, value, flag)`
   - Retrieve via `get_state(name)`
   - Never directly access `wrapper_states` or set fields.

3. **Call `super().__init__(optimizer, max_draws)`** before subclass setup

**Subclasses may:**
- Implement any control algorithm (fixed accumulation, quality thresholds, rescaling, etc.)
- Fiddle with the gradients. Keep in mind the sum of gradients are provided, not the average, in the .grad accumulators as averaging is done as part of taking the optimizer steps.

Attempts to initialize and set fields on the class directly will throw, and in fact must go through set_state instead. set_state lets you set a statistic as 'vital', and one should do so with important fields like, for example, num_last_batch_draws or last_mean_gradient_norm. get_state can get optimizer state as well, and when doing so will return a list rather than a single entry from the optimizer parameter groups; a special flag can ask for the mean, max, or min instead.


## Essential Details: Essential technical facts for implementers

**Counters: Found in .get_state**
- `num_batches` - Total batches processed since wrapper creation
- `num_steps` - Total optimizer steps taken
- `num_draws` - Batches accumulated since last step (resets to 0 after each step)

**Gradient Accumulation:**
PyTorch's `.backward()` sums gradients into `.grad` by default. After N consecutive backward passes, each parameter's `.grad` contains the sum of N batches. The wrapper exploits this: `_take_optimizer_step()` multiplies all gradients by `1/N` to convert sum → mean before stepping. 

**State Storage:**
All wrapper state lives in `wrapper_states`: `{name: {"value": X, "flag": "vital"/"optional"}}`. Access only through `set_state()` and `get_state()`. Direct field assignment after initialization throws to prevent serialization bugs - if it's not in `wrapper_states`, it won't survive `state_dict()`/`load_state_dict()`.

## Method Specifications

### Constructor

**Subclasses must call via `super().__init__(optimizer, max_draws)` before their own initialization.**

The parent constructor sets up the base wrapper infrastructure. When your subclass calls `super().__init__()`, it wraps the provided optimizer, initializes the counter system (num_batches, num_steps, num_draws all to 0), and stores the max_draws safety bound. After this call returns, your subclass can use `set_state()` to initialize algorithm-specific parameters, knowing the base wrapper infrastructure is ready.

The optimizer parameter should be a fully-configured PyTorch optimizer (Adam, SGD, etc.) with learning rate, weight decay, and other hyperparameters already set. The wrapper doesn't modify optimizer configuration - it just wraps it for gradient accumulation control.

```python
def __init__(self, optimizer: torch.optim.Optimizer, max_draws: int)
```

**Parameters:**
- `optimizer` (torch.optim.Optimizer) - Configured PyTorch optimizer to wrap
- `max_draws` (int) - Maximum batches that can accumulate before forcing a step. Must be ≥ 1.

**Initializes:**
- Wraps the optimizer for transparent delegation
- Initializes counters: `num_batches=0`, `num_steps=0`, `num_draws=0`
- Stores `max_draws` as safety bound

**Subclass Implementation Pattern:**
```python
class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    def __init__(self, optimizer, max_draws, threshold):
        super().__init__(optimizer, max_draws)  # Call parent first

        # Now initialize subclass-specific state
        self.set_state("gradient_norm_threshold", threshold, "vital")
        self.set_state("last_grad_norm", None, "optional")
```

### step (Abstract)

**Must be implemented by subclasses.**

The step method is where control algorithm logic lives. In standard PyTorch training, you call `optimizer.step()` after every batch. With wrappers, the training loop is unchanged - you still call `step()` after every batch - but now the wrapper intercepts to make a binary decision: actually step the optimizer, or continue accumulating gradients. This is the Sequential Binary Decision Controller pattern in action. Exact decision formula varies.

```python
def step(self) -> bool
```

**Returns:**
- `bool` - True if optimizer stepped this call, False if still accumulating

**Implementation Pattern:**

When implementing, you must call `_batch_received()` first to update counters, then implement your decision logic, then call `_take_optimizer_step()` if you decide to step. Return True if you stepped, False if accumulating.

```python
def step(self):
    self._batch_received()  # Always first - updates counters

    # Decision logic here - examine gradients, metrics, schedules, etc.
    if should_step:
        self._take_optimizer_step()
        return True
    return False
```

- Call `_batch_received()` exactly once at method start
- Call `_take_optimizer_step()` when deciding to step
- Return True/False indicating whether optimizer stepped
- Called once per training batch

### statistics

The statistics method provides complete visibility into the wrapper's internal state and the underlying optimizer's configuration. Call this when you need to log detailed training metrics, debug unexpected behavior, analyze algorithm performance, or export comprehensive training data. It returns everything - all wrapper state (both vital and optional), all float optimizer hyperparameters, and all counters. This is the "full dump" for comprehensive monitoring, analysis, or debugging.

The method pulls from two sources: wrapper state (counters, algorithm parameters, cached metrics) and optimizer parameter groups (lr, weight decay, momentum, etc.). For optimizers with multiple parameter groups where a value differs across groups (e.g., different learning rates), the `aggregate_behavior` parameter controls aggregation method. The aggregated value gets a `*` suffix added to the key to signal heterogeneity. If all groups have the same value, it returns that value without suffix.

This method is read-only and deterministic - calling it never modifies state, and the same state always produces the same output. You can call it as many times as needed, even before the first step.


```python
def statistics(self, aggregate_behavior: Literal["mean", "max", "min"] = "mean") -> Dict[str, Any]
```

**Parameters:**
- `aggregate_behavior` (Literal["mean", "max", "min"]) - How to aggregate multi-group optimizer parameters. Default: "mean".

**Returns:**
- `Dict[str, Any]` - Complete statistics dictionary

### vital_statistics

The vital_statistics method returns a curated subset of statistics designed for real-time monitoring during training. Call this to populate progress bars (tqdm), training dashboards, or logging systems where you need to track key health metrics without overwhelming the display. It's the "health dashboard" - just the metrics that matter for monitoring training progress and diagnosing problems at a glance.

Unlike `statistics()` which returns everything, this method filters to only vital metrics - those marked with `flag="vital"` when stored via `set_state()`. Subclasses decide what's vital: counters like num_batches and num_draws are always vital, and algorithm-specific metrics like gradient norms or quality thresholds should be marked vital if they're key performance indicators. The method also includes optimizer hyperparameters (lr, weight decay, etc.) since these are essential for understanding training behavior, especially with schedulers. The `aggregate_behavior` parameter controls how multi-group optimizer parameters are aggregated, with a `*` suffix for heterogeneous values.

This method is read-only, deterministic, and safe to call frequently. It's a strict subset of `statistics()` - every key in vital_statistics also appears in statistics.

```python
def vital_statistics(self, aggregate_behavior: Literal["mean", "max", "min"] = "mean") -> Dict[str, Any]
```

**Parameters:**
- `aggregate_behavior` (Literal["mean", "max", "min"]) - How to aggregate multi-group optimizer parameters. Default: "mean".

**Returns:**
- `Dict[str, Any]` - Curated vital statistics dictionary

**Contents:**
- wrapper_states entries marked `flag="vital"`
- `num_batches` and `num_draws` (always vital)
- All float-valued keys from `optimizer.param_groups`
- Multi-group aggregation: same rules as `statistics()`

**Properties:**
- Read-only - never modifies state
- Deterministic - same state produces same output
- Strict subset of `statistics()`
- Works before first step

**Use Cases:**
```python
from tqdm import tqdm

pbar = tqdm(train_loader)
for batch in pbar:
    # ... forward, backward ...
    stepped = optimizer.step()

    pbar.set_postfix(optimizer.vital_statistics())  # Real-time monitoring
```

### state_dict

```python
def state_dict(self) -> Dict[str, Any]
```

The state_dict method serializes the complete wrapper state for checkpointing. Call this periodically during training to save checkpoints - if training crashes, gets preempted, or you want to resume later, you can restore to this exact point. This is PyTorch's standard checkpointing pattern extended to wrappers. The method captures everything needed for lossless resumption: all wrapper state (counters, algorithm parameters, cached metrics), the complete optimizer state (momentum buffers, adaptive learning rate state, etc.), and any other internal state.

The guarantee is strong: if you call `state = wrapper.state_dict()`, save it, restart the process, create a new wrapper with the same configuration, and call `wrapper.load_state_dict(state)`, training resumes exactly where it left off. No observable difference from never stopping - same counters, same accumulated gradients (via num_draws), same optimizer momentum, same everything. This works because the wrapper stores all state through `set_state()` which writes to `wrapper_states`, and state_dict dumps both `wrapper_states` and `optimizer.state_dict()`.

**Returns:**
- `Dict[str, Any]` - Complete serialized state

**Preserves:**
- All `wrapper_states` (vital and optional entries)
- Complete `optimizer.state_dict()` (optimizer state, momentum, etc.)
- All counters (num_batches, num_steps, num_draws)
- All cached values and algorithm parameters

**Contract:**
Lossless resumption - `state_dict()` → save → restart → `load_state_dict()` → training continues exactly as if never interrupted.

**Usage Pattern:**
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pt')

# Later, resume:
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

### load_state_dict

```python
def load_state_dict(self, state_dict: Dict[str, Any]) -> None
```

The load_state_dict method restores wrapper state from a checkpoint created by `state_dict()`. Call this after creating a new wrapper instance to resume training from a saved checkpoint. This is the restore half of PyTorch's checkpointing pattern - you create a fresh wrapper with the same configuration (same optimizer type, same max_draws), then load the saved state to pick up exactly where training left off.

The method restores everything: wrapper state (counters, algorithm parameters, cached values), optimizer state (momentum buffers, learning rate state, parameter-specific state), and all internal structures. After loading, the wrapper behaves identically to the wrapper that created the checkpoint - same num_batches count, same num_draws accumulation state, same everything. You can continue training seamlessly.

The typical pattern: save checkpoints periodically during training, and on startup (or after crash), check if a checkpoint exists and load it. This provides fault tolerance and allows pausing/resuming training.

**Parameters:**
- `state_dict` (Dict[str, Any]) - State dictionary from `state_dict()` method

**Contract:**
Restores wrapper to exact state when `state_dict()` was called. Training continues seamlessly with no observable difference from never stopping.

**Usage Pattern:**
```python
# At startup - resume if checkpoint exists
if os.path.exists('checkpoint.pt'):
    checkpoint = torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['wrapper'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0
```

### zero_grad


The zero_grad method is intentionally disabled and always raises an error. This is a safety mechanism to prevent a common mistake when migrating to optimizer wrappers. In standard PyTorch training, you call `optimizer.zero_grad()` at the start of each batch. With wrappers, this pattern breaks gradient accumulation - the wrapper needs gradients to accumulate across multiple batches, and clearing them would corrupt the accumulation logic.

The wrapper handles gradient clearing automatically inside `_take_optimizer_step()` - after averaging and stepping, it zeros gradients. You never manually zero gradients with wrappers. This method exists solely to fail fast with a clear error message instead of silently corrupting training when someone forgets and calls `zero_grad()` out of habit.

```python
def zero_grad(self) -> None
```
**Raises:** `NotImplementedError` - Always.

**Migration Note:**
```python
# Old pattern (standard optimizer):
for batch in loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()

# New pattern (with wrapper):
for batch in loader:
    # No zero_grad() call!
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Wrapper handles everything
```

### set_state (For Subclasses)

```python
def set_state(self, name: str, value: Any, flag: Literal["vital", "optional", "optimizer"]) -> None
```

**For subclass implementation - stores wrapper state and exposes scheduler targets.**

The set_state method serves two purposes: storing wrapper state and exposing parameters to ScheduleAnything for dynamic scheduling. Call this in your `__init__` to initialize state and register schedulable parameters, and in your `step()` implementation to update state as training progresses. This is the only way to store state in subclasses - direct field assignment (e.g., `self.threshold = 0.5`) is prohibited after initialization and will throw an error.

Why the restriction? Serialization guarantee. All state stored through `set_state()` goes into `wrapper_states`, which `state_dict()` automatically serializes. If you set fields directly, they won't be in `wrapper_states` and won't survive checkpointing. The wrapper enforces this at runtime to prevent silent serialization bugs that only show up when you try to resume training. Since this is for subclassing, which happens infrequently, it is deemed acceptable.

The `flag` parameter controls three distinct behaviors:

- **`"vital"`**: Stores in wrapper_states, appears in `vital_statistics()`, serializes
- **`"optional"`**: Stores in wrapper_states, only in `statistics()`, serializes
- **`"optimizer"`**: Uses ScheduleAnything to extend the wrapped optimizer with this schedulable parameter. The parameter becomes accessible to schedules just like native optimizer parameters (lr, weight_decay). Does NOT store in wrapper_states - the optimizer owns it.

The "optimizer" flag enables wrapper-specific scheduling. For example, OptimizerWrapperGNTS needs to schedule `gradient_norm_threshold` alongside learning rate. By calling `set_state("gradient_norm_threshold", 0.95, "optimizer")`, the threshold becomes a schedulable parameter in the optimizer's param_groups, accessible to ScheduleAnything schedules. Schedules can then bind to it just like they bind to lr.

Once set, the flag is immutable - attempting to change it throws an error to prevent inconsistencies.

**Parameters:**
- `name` (str) - State variable name. For "vital"/"optional": cannot collide with optimizer param_group keys. For "optimizer": becomes a param_group key.
- `value` (Any) - Value to store. Must be serializable for "vital"/"optional". Must be numeric for "optimizer".
- `flag` (Literal["vital", "optional", "optimizer"]) - Storage and scheduling behavior. Required, no default.

**Contract:**
- `flag="vital"/"optional"`: Stores in `wrapper_states` dictionary, serializes
- `flag="optimizer"`: Extends optimizer via ScheduleAnything, adds to `.valid_schedule_targets`
- Creates new entry on first call for this name
- Updates value on subsequent calls (flag must match original)
- Throws if flag="vital"/"optional" name collides with optimizer param_group keys
- Throws if flag differs from original flag for this name

**Usage Pattern:**
```python
class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    def __init__(self, optimizer, max_draws, threshold):
        super().__init__(optimizer, max_draws)

        # Wrapper state - internal tracking
        self.set_state("last_grad_norm", None, "optional")
        self.set_state("num_last_batch_draws", 0, "vital")

        # Schedulable parameter - exposed to ScheduleAnything
        self.set_state("gradient_norm_threshold", threshold, "optimizer")
        # Now schedules can bind to "gradient_norm_threshold" like "lr"
```

### get_state (Unified Access)


**Purpose:**
Transparent access to both wrapper state (`num_batches`, custom thresholds) and optimizer state (`lr`, `weight_decay`) through one interface. Simplifies subclass implementation by eliminating manual source checking.

The get_state method retrieves state from either the wrapper or the underlying optimizer through a single unified interface. Call this in your `step()` implementation to access algorithm parameters, counters, cached metrics, or optimizer hyperparameters. Instead of checking "is this in wrapper_states or optimizer.param_groups?", you just call `get_state()` and it searches both.

This unified access pattern is particularly useful for subclass implementations that need to examine both wrapper state (like gradient norms or batch counts) and optimizer state (like current learning rate or weight decay). The method searches wrapper_states first, then falls back to optimizer.param_groups. For multi-group optimizers where a parameter differs across groups, the `aggregate_behavior` parameter controls how to aggregate: return the mean (default), max, min, or the raw list of values.

```python
def get_state(self, name: str, aggregate_behavior: Optional[Literal["none", "mean", "max", "min"]]) -> Any
```

**Parameters:**
- `name` (str) - State variable name to retrieve
- `aggregate_behavior` (Optional[Literal["none", "mean", "max", "min"]]) - How to aggregate multi-group optimizer parameters. 
- 
**Returns:**
- `Any` - The state value (just the value, not metadata). For multi-group params, returns aggregated value or list.

**Contract - Search Order:**
1. If `name` in `wrapper_states`: return that value (ignore aggregate_lists)
2. Else if `name` in `optimizer.param_groups`:
   - All groups same value → return that value
   - Groups have different values:
     - `aggregate_lists="mean"` → return mean
     - `aggregate_lists="max"` → return max
     - `aggregate_lists="min"` → return min
     - `aggregate_lists="none"` → return list of values
3. Else: throw error (not found)

### _batch_received (Protected)

```python
def _batch_received(self) -> None
```

**Purpose:**
Centralized counter management to maintain invariants consistently across all subclass implementations.

For subclasses - required to use.

The _batch_received method handles centralized counter updates when a batch is processed. Call this at the start of your `step()` implementation, before any decision logic. The method increments `num_batches` (total batches processed since creation) and `num_draws` (batches accumulated since last optimizer step). It also enforces the `max_draws` safety bound - if calling this would push num_draws beyond max_draws, it throws an error to prevent unbounded accumulation.

Why centralized counter management? Consistency and invariant enforcement. If every subclass manually incremented counters, bugs would slip through - forgetting to increment, incrementing twice, not checking max_draws, etc. Centralizing this logic in one method that all subclasses must call ensures counters stay correct and invariants hold. The method uses `set_state()` to update counters, so they automatically serialize and appear in statistics.

Subclasses must call this exactly once per batch, typically as the first line of `step()`. Calling it zero times means counters don't advance (broken statistics). Calling it multiple times per batch means counters advance too fast (broken invariants). The base class can't call it automatically because only the subclass knows when a batch has been processed.

### _take_optimizer_step (Protected)

**Purpose:**
Ensures gradient averaging invariant maintained consistently. Subclasses **must** use this to step - bypassing it violates the contract and produces incorrect training behavior.

The _take_optimizer_step method encapsulates the entire gradient averaging and optimizer stepping process. Call this in your `step()` implementation when your decision logic determines it's time to step the optimizer. This is the only way to step the optimizer - directly calling `self.optimizer.step()` bypasses gradient averaging and breaks the wrapper's contract.

Why must subclasses use this? Gradient averaging guarantee. PyTorch's `.backward()` sums gradients across multiple backward passes. If you've accumulated N batches (num_draws = N), each parameter's `.grad` contains the sum of N batches' gradients. This method divides all gradients by N to convert sum → mean, then steps the optimizer with the averaged gradients. After stepping, it zeros gradients for the next accumulation cycle, increments counters, and caches the gradient norm for statistics.

The method executes a precise sequence: average gradients, compute L2 norm, step optimizer, zero gradients, update counters and state. This sequence ensures the gradient averaging invariant holds consistently. If subclasses step the optimizer directly, they skip averaging and the optimizer sees summed gradients instead of means - wrong effective batch size, broken training dynamics. The method also throws if num_draws is 0 (no batches accumulated yet), preventing invalid steps.


```python
def _take_optimizer_step(self) -> None
```

**Contract**
1. Average gradients: multiply all by `1 / num_draws`
2. Compute L2 gradient norm across all parameters; store as "last_grad_norm"
3. Step wrapped optimizer with averaged gradients
4. Zero all gradients
5. Update state: store gradient norm, increment `num_steps`, reset `num_draws` to 0

**Throws:**
- If `num_draws == 0` (cannot step without accumulated batches)

## Invariants

Properties that **must always hold:**

1. **`num_batches >= num_steps`** - Cannot step more than batches processed
2. **`num_draws <= max_draws`** - Cannot exceed maximum accumulation
3. **Gradient averaging** - Gradients always averaged before stepping
4. **One batch minimum** - At least one batch before any step
5. **State preservation** - `state_dict()` → `load_state_dict()` preserves exact state
6. **Transparent forwarding** - Non-overridden methods/attributes forward to `optimizer`
7. **Required utilities** - Subclasses must use `_batch_received()` and `_take_optimizer_step()`
8. **Namespace separation** - `set_state()` cannot use optimizer param_group names
9. **Immutable flags** - Vital/optional status cannot change once set
10. **Anytime statistics** - `statistics()` and `vital_statistics()` work before first step

## Subclass Implementation Contract

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

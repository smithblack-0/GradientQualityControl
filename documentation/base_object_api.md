# Optimizer Wrapper Base class

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

The base class solves these problems once. It manages gradient accumulation by dividing accumulated gradients by the number of batches when stepping the optimizer, converting PyTorch's natural summation into the required mean. It tracks counters through `_received_batch()` and `_take_optimizer_step()`, maintaining invariants like ensuring at least one batch is drawn before any step and preventing accumulation beyond `max_draws`. It provides serialization through `state_dict()` and `load_state_dict()`, automatically saving and restoring all wrapper state alongside the wrapped optimizer's state. Subclasses store their algorithm parameters through `_set_state()`, which writes to an internal dictionary that serializes automatically - direct field assignment after initialization throws an error to prevent serialization bugs. The base class handles reporting through `statistics()` and `vital_statistics()`, pulling from both wrapper state and optimizer parameters and aggregating across parameter groups when values differ. It implements transparent optimizer interface forwarding by subclassing `torch.optim.Optimizer` and delegating all non-overridden methods and attributes to the wrapped optimizer.

The base class does not implement the control algorithm. Subclasses implement `step()` to decide when conditions warrant calling `_take_optimizer_step()`. The base class provides infrastructure; subclasses provide control logic. Overall, the user experience should be that there is a fairly normal optimizer that just calls .zero_grad itself.

### Distributed Contract

- Provide distributed mode when needed
- Do not bind threads to only a subset of available worlds. 

### ScheduleAnything Integration

Optimizer wrappers integrate seamlessly with ScheduleAnything to enable dynamic scheduling of both optimizer parameters (lr, weight_decay) and wrapper-specific control parameters (gradient_norm_threshold, logical_batch_size, etc.). This integration happens through parameter extension and unified access.

**How it works:**

When a subclass calls `_set_state(name, value, "optimizer")`, the base class uses ScheduleAnything to extend the wrapped optimizer's param_groups with that new parameter. The parameter gets injected into every parameter group alongside native optimizer parameters like lr. After extension, ScheduleAnything schedules can bind to and modify this parameter just like they would lr or weight_decay.

For example, OptimizerWrapperGNTS needs to schedule `gradient_norm_threshold`. During `__init__`, it calls `_set_state("gradient_norm_threshold", 1.0, "optimizer")`. This extends the AdamW optimizer to include gradient_norm_threshold in its param_groups. A ScheduleAnything schedule can then bind to both lr and gradient_norm_threshold, warming up lr while annealing the threshold. 

The optimizer wrapper exposes `.valid_schedule_targets` to list all schedulable parameters. This includes the optimizer's native parameters plus any wrapper-specific parameters added via the "optimizer" flag. Factory functions use this to verify schedule bindings are valid.

### Public Instance Behavior

Instances of this object pretend to be, almost completely, an optimizer instance. This means the object is implemented as on optimizer subclass, and downstream code can invoke the .step field as normal, at which point the underlying subclasses's step algorithm is used to judge whether or not to take a step.  

The following fields are available, with indicated behavior, on the main class

**Public Fields**:
- **`optimizer`** (`torch.optim.Optimizer`) - The wrapped optimizer, directly accessible

**Public Properties**:
- **`num_batches`** (`int`) - Total batches processed since wrapper creation
- **`num_steps`** (`int`) - Total optimizer steps taken
- **`num_draws`** (`int`) - Batches accumulated since last step (resets to 0 after each step)
- **`last_num_draws`** (`int`) - Number of batches in most recent optimizer step. None before first step.
- **`last_grad_norm`** (`float`) - L2 gradient norm from most recent optimizer step. None before first step.
- **`valid_schedule_targets`** (`List[str]`) - Read-only list of all schedulable parameter names. Includes native optimizer parameters (lr, weight_decay, momentum, etc.) and wrapper-extended parameters.
- **`distributed_mode`**: (Optional[Litera["replicated", "sharded"]]). A specification of the kind of distributed mode. 
- **`device`**: The device the optimizer is on. 

**Public Methods**
- **`step`**: Transparently duck-types exactly as before. 
- **`zero_grad`**: Now throws. The wrapper resets grads instead.
- **`statistics`**: Returns dictionary of features governing internal behavior.
- **`vital_statistics`**: Returns dictionary of things to display or log that are vital performance indicators; subclasses must judge and mark whether something is vital.
- **`state_dict`**: Gets the state dict, storing the wrapped optimizer and the state dict from the layer.
- **`load_state_dict`**: Losslessly resumes from a state dict.

**Optimizer Mocking**

From an external perspective, this IS the wrapped optimizer, not a wrapper. All other methods or calls are automatically and transparently passed into the wrapped optimizer. If you call, for instance, the param_groups field you get the group off the base optimizer.

### Subclassing Instance Behavior

Using the subclass to implement an algorithm requires knowledge of a few contract details. First, and perhaps most important, is what a subclass should usually implement. Under standard conditions, you are responsible for implementing only one thing: step.

**Subclasses must:**

1. **Implement** Metric retrieval strategies as methods on the subclass; distributed reduction static methods for "sharded" and "replicated" on the subclass.
2. **Bind** Metrics to `._get_metric` using `._bind_metric` thus configuring it for distributed usage. This involves configuring the reduction behavior for sharded and replicated distribution cases.
3 **Implement `step() -> bool`**
   - Use `._get_metric(*args, **kwargs)` as appropriate to get bound metrics which handle distributed quirks. 
   - Return True/False for stepped/accumulating
   - Call `_batch_received()` once per batch
   - Call `_take_optimizer_step()` when stepping

2. **Use state management**
   - Store all state via `_set_state(name, value, flag)`
   - Retrieve via `_get_state(name, aggregation_behavior)`
   - Setting fields directly will instead throw.

3. **Call `super().__init__(optimizer, max_draws, distributed_mode)`** before subclass setup

**Subclasses may:**
- Implement any control algorithm.
- Fiddle with the gradients. Keep in mind the sum of gradients are provided, not the average, in the .grad accumulators as averaging is done as part of taking the optimizer steps.

## Essential Details: Essential technical facts for implementers

All properties below are accessible as direct attributes and through `_get_state()`:

- `num_batches` - Total batches processed. Increments on each `_batch_received()` call. **Vital** 
- `num_steps` - Total optimizer steps taken. Increments on each `_take_optimizer_step()` call.  **Vital**.
- `num_draws` - Batches accumulated since last step. Increments on `_batch_received()`
- `last_num_draws` - Cached value of `num_draws` from most recent step.  **Vital**.
- `last_grad_norm` - Cached L2 gradient norm from most recent step. **Vital**.

**Vital** means the property appears in `vital_statistics()`.

## Method Specifications

### Constructor

The parent constructor sets up the base wrapper infrastructure. When your subclass calls `super().__init__()`, it wraps the provided optimizer, initializes the counter system (num_batches, num_steps, num_draws all to 0), stores the max_draws safety bound, and sets the distributed flag for subclasses. After this call returns, your subclass can use `_set_state()` to initialize algorithm-specific parameters, knowing the base wrapper infrastructure is ready.

The optimizer parameter should be a fully-configured PyTorch optimizer (Adam, SGD, etc.) with learning rate, weight decay, and other hyperparameters already set. The wrapper doesn't modify optimizer configuration - it just wraps it for gradient accumulation control.

```python
def __init__(self, optimizer: torch.optim.Optimizer, 
             max_draws: int = 64,
             distributed_mode: Optional[Literal["replicated", "sharded"]] = None)
```

**Parameters:**
- `optimizer` (torch.optim.Optimizer) - Configured PyTorch optimizer to wrap
- `max_draws` (int = 64) - Maximum batches that can accumulate before forcing a step. Must be ≥ 1.
- `distributed_mode` (Optional[Literal["replicated", "sharded"]] = None) - Distributed training mode for metric aggregation. Stored for subclass use via `_get_state("distributed_mode")`. Replicated mode is for data-parallel (DDP), sharded mode is for model-parallel.

**Raises**:

- `TypeError`: On clear mismatch for optimizer, max_draws, distributed mode
- `RuntimeError`: If a distributed pool is noticed but the distributed mode is not being set.

**Initialization**

Sets up initial state. Exact details will depend on implementation, however num_draws, num_steps, and num_batches must contractually be accurate at all times. Presumably sets a few fields too. Exact details should be inferred by rest of contract and implementation strategy, or checked in source. 

**Subclassing Pattern:**

Subclassing should first initialize the system in general, then 
set optional, optimizer, or mandatory state and can retrieve that
state later in step. 

```python
class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    def __init__(self, optimizer, max_draws=64, distributed_mode=None):
        super().__init__(optimizer, max_draws, distributed_mode)  # Call parent first

        # Now initialize subclass-specific state
        self._set_state("gradient_norm_threshold", 1.0, "optimizer")  # Schedulable

        # Subclass can access distributed_mode via _get_state()
        distributed_mode = self._get_state("distributed_mode")
```

**Binding Metrics**

To support distributed modes of operation, the user needs to specify both how to take metrics, and how to reduce them when multiple devices are involved. This typically means implementing a set of three functions per metric to be used, then binding them. Consult the utilities for useful prebuilts. Keep in mind the right action often varies between "replicated" and "sharded" modes. For instance, you need to average across replicated models, but add across sharded modes. 

Note that what a "metric" is can be interpreted creatively. For instance, an excellent way to find the overall batch size if it matters is to sum up individual physical batch sizes when replicated. The thing need not be a traditional metric. 


```python
class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    
    def get_gradient_norm(self) -> float:
        """Just a L2 grad norm."""
        # This could have had arguments if desired.
        all_params = [p for group in self.optimizer.param_groups for p in group['params']]
        grads = [p.grad for p in all_params if p.grad is not None]
        return torch.nn.utils.get_total_norm(grads)
    
    def merge_sharded_grad_norms(self, grad_norm: float)->float:
        """L2 Sum is required for grad norms"""
        # implements sqrt(sum(square(input))) across devices
        grad_norm_tensor = torch.tensor([grad_norm], device=self.device)  
        grad_norm_tensor = grad_norm_tensor**2
        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
        grad_norm_tensor = torch.sqrt(grad_norm_tensor)
        return grad_norm_tensor.item()
    
    def merge_replicated_grad_norms(self, grad_norm: float)->float:
        """L2 (RMS) mean is required to merge"""
        # implements sqrt(sum(square(input))/num_devices) across devices
        num_devices = dist.get_world_size()
        grad_norm_tensor = torch.tensor([grad_norm], device=self.device)  
        grad_norm_tensor = grad_norm_tensor**2
        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
        grad_norm_tensor = grad_norm_tensor/num_devices
        grad_norm_tensor = torch.sqrt(grad_norm_tensor)
        return grad_norm_tensor.item()
        
    
    def __init__(self, optimizer, max_draws=64, distributed_mode=None):
        super().__init__(optimizer, max_draws, distributed_mode)  # Call parent first


        # Subclass can access distributed_mode via _get_state()
        distributed_mode = self._get_state("distributed_mode")

        # Subclasses can now get "grad_norm" using ._get_metric("grad_norm"). 
        self.bind_metric("grad_norm", self.get_gradient_norm,
                         self.merge_replicated_grad_norms, self.merge_sharded_grad_norms )

    def step(self):
        # Get the bound metric. Handle distributed logic. 
        grad_norm = self.get_metric("grad_norm")
```




### statistics

The statistics method provides complete visibility into the wrapper's internal state and the underlying optimizer's configuration. Call this when you need to log detailed training metrics, debug unexpected behavior, analyze algorithm performance, or export comprehensive training data. 

The method pulls from vital, optional, and optimizer information sources, and a whitelist filter can be defined in terms of "vital", "optional", and "optimizer" to isolate relevant regions. By default, with None, all entries are returned. 

Some state cases, and optimizers, present a unique challenge; they can have multiple param groups with different values. If all values are the same information is displayed like normal. If they are not, the aggregated value gets a "*" next to it, like "lr*" and the `aggregate_behavior` flag governs how this is reduced suffix. Only optimizer fields that are of float or scalar-tensor type are isolated and displayed in this manner. This may also occur when storing, for example, a list of states as subclass state.

This method is read-only and deterministic - calling it never modifies state, and the same state always produces the same output. You can call it as many times as needed, even before the first step.


```python
def statistics(self,
               behavior: List[Literal["vital", "verbose"]] = "verbose",
               aggregate_behavior: Literal["mean", "max", "min"] = "mean",
               ) -> Dict[str, Any]:
```

**Parameters:**
- `behavior` You may choose between vital and verbose. Verbose includes chosen optimizer state, optional fields, and vital state. Vital only includes optimizer state and vital.
- `aggregate_behavior` (Literal["mean", "max", "min"]) - How to aggregate multi-group optimizer parameters. Default: "mean".

**Returns:**
- `Dict[str, Any]` - Statistics dictionary.

**Contents:**
In  default verbose mode

- All built-in properties (see Essential Details)
- All wrapper_states entries (vital and optional)
- All float-valued keys from `optimizer.param_groups` (lr, weight_decay, etc.)
- Multi-group parameters: aggregated per `aggregate_behavior`, with `*` suffix if heterogeneous

### vital_statistics

The vital_statistics method returns a curated subset of statistics designed for real-time monitoring during training. Call this to populate progress bars (tqdm), training dashboards, or logging systems where you need to track key health metrics without overwhelming the display. It's the "health dashboard" - just the metrics that matter for monitoring training progress and diagnosing problems at a glance. It is implemented as an alias into statistics that calls with 'vital'.

```python
def vital_statistics(self, aggregate_behavior: Literal["mean", "max", "min"] = "mean") -> Dict[str, Any]
```

**Parameters:**
- `aggregate_behavior` (Literal["mean", "max", "min"]) - How to aggregate multi-group optimizer parameters. Default: "mean".

**Returns:**
- `Dict[str, Any]` - Curated vital statistics dictionary

**Contents:**
- Most built-in properties. 
- wrapper_states entries marked `flag="vital"`
- All float-valued or scalar tensor keys from `optimizer.param_groups`

**Use Cases:**

Excellent for tqdm or logging. 

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
def state_dict(self) -> Dict[str, Any]:
```

The state_dict method serializes the complete optimizer wrapper state for checkpointing. Call this periodically during training to save checkpoints - if training crashes, gets preempted, or you want to resume later, you can restore to this exact point. This is PyTorch's standard checkpointing pattern extended to wrappers. The method captures everything needed for lossless resumption: all wrapper state (counters, algorithm parameters, cached metrics), the complete optimizer state (momentum buffers, adaptive learning rate state, etc.), and any other internal state.

**Returns:**
- `Dict[str, Any]` - Complete serialized state

**Preserves:**
- All `wrapper_states` (vital and optional entries)
- Complete `optimizer.state_dict()` (optimizer state, momentum, etc.)
- All internal state, as it is stored in wrapper_states

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

The load_state_dict method restores optimizer wrapper state from a checkpoint created by `state_dict()`. Call this after creating a new optimizer instance to resume training from a saved checkpoint. This is the restore half of PyTorch's checkpointing pattern - you create a fresh wrapper with the same configuration (same optimizer type, same max_draws), then load the saved state to pick up exactly where training left off.

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

```python
def zero_grad(self) -> None:
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

## Subclass Responsibility

### step (Abstract)

**Must be implemented by subclasses.**

The step method is where control algorithm logic lives. In standard PyTorch training, you call `optimizer.step()` after every batch. With wrappers, the training loop is unchanged - you still call `step()` after every batch - but now the wrapper intercepts to make a binary decision: actually step the optimizer, or continue accumulating gradients. This is the Sequential Binary Decision Controller pattern in action. Exact decision formula varies.

```python
def step(self, *args, **kwargs) -> bool:
```

**Parameters:**
- `*args, **kwargs` - Optional arguments for control algorithms requiring additional inputs (e.g., OptimizerWrapperMHT requires metric value). Most wrappers ignore these.

**Returns:**
- `bool` - True if optimizer stepped this call, False if still accumulating

**Subclassing Pattern:**

When subclassing, you must call `_batch_received()` first to update counters, then implement your decision logic, then call `_take_optimizer_step()` if you decide to step. Return True if you stepped, False if accumulating. The canonical strategy is shown below.

```python
def step(self, *args, **kwargs):
    self._batch_received()  # Always first - updates counters

    # Decision logic here - examine gradients, metrics, schedules, etc.
    # Can extract needed args/kwargs if algorithm requires them
    if should_step:
        self._take_optimizer_step()
    return should_step
```

- Call `_batch_received()` exactly once at method start
- Call `_take_optimizer_step()` when deciding to step
- Return True/False indicating whether optimizer stepped
- Called once per training batch

### _set_state (For Subclasses)

```python
def _set_state(self, name: str, value: Any, flag: Literal["vital", "optional", "optimizer"]) -> None
```

**For subclass implementation - stores wrapper state and exposes scheduler targets.**

The _set_state method serves two purposes: storing wrapper state and exposing parameters to ScheduleAnything for dynamic scheduling. Call this in your `__init__` to initialize state and register schedulable parameters, and in your `step()` implementation to update state as training progresses. This is the only way to store state in subclasses - direct field assignment (e.g., `self.threshold = 0.5`) is prohibited after initialization and will throw an error.

The "optimizer" flag enables wrapper-specific scheduling. For example, OptimizerWrapperGNTS needs to schedule `gradient_norm_threshold` alongside learning rate. By calling `_set_state("gradient_norm_threshold", 0.95, "optimizer")`, the threshold becomes a schedulable parameter in the optimizer's param_groups, accessible to ScheduleAnything schedules. Schedules can then bind to it just like they bind to lr.

Once set, the flag is immutable - attempting to change it throws an error to prevent inconsistencies.

**Parameters:**
- `name` (str) - State variable name. For "vital"/"optional": cannot collide with optimizer param_group keys. For "optimizer": becomes a param_group key.
- `value` (Any) - Value to store. Must be serializable for "vital"/"optional". Must be float for "optimizer".
- `flag` (Literal["vital", "optional", "optimizer"]) - Storage and scheduling behavior. Required, no default.

**Contract:**
- `flag="vital"/"optional"`: Stores in `wrapper_states` dictionary, serializes
- `flag="optimizer"`: Extends optimizer via ScheduleAnything, adds to `.valid_schedule_targets`
- Creates new entry on first call for this name
- Updates value on subsequent calls (flag must match original)
- Throws RuntimeError if flag="vital"/"optional" name collides with optimizer param_group keys
- Throws RuntimeError if flag differs from original flag for this name

**Usage Pattern:**

Usage is legal in initialization or step.

```python
class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    def __init__(self, optimizer, max_draws, threshold):
        super().__init__(optimizer, max_draws)
        
        self._set_state("my_special_control_feature", 1.0, "optional")

        # Schedulable parameter - exposed to ScheduleAnything
        self._set_state("gradient_norm_threshold", threshold, "optimizer")
        # Now schedules can bind to "gradient_norm_threshold" like "lr"
    def step(self):
        ...
        control = self._get_state("my_special_control_feature")
        self._set_state("my_special_control_feature", control, "optional")

`
```

### _get_state (Unified Access)

Transparent access to both wrapper state (`num_batches`, custom thresholds) and optimizer state (`lr`, `weight_decay`) through one interface. Simplifies subclass implementation by eliminating manual source checking.

The _get_state method retrieves state from either the wrapper or the underlying optimizer through a single unified interface. Call this in your `step()` implementation to access algorithm parameters, counters, cached metrics, or optimizer hyperparameters.

This unified access pattern is particularly useful for subclass implementations that need to examine both wrapper state (like gradient norms or batch counts) and optimizer state (like current learning rate or weight decay). The method searches wrapper_states first, then falls back to optimizer.param_groups. For optimizer parameters, the `aggregate_behavior` parameter controls whether you receive a list of values or an aggregated scalar.

```python
def _get_state(self,
               name: str,
               aggregate_behavior: Optional[Literal["mean", "max", "min"]] = None,
               ) -> Any:
```

**Parameters:**
- `name` (str) - State variable name to retrieve
- `aggregate_behavior` (Optional[Literal["mean", "max", "min"]]) - How to aggregate optimizer parameters. If None, returns list. If mean/max/min, returns aggregated scalar. Default: None.

**Returns:**
- `Any` - The state value. Wrapper states return stored value directly. Optimizer parameters return list (if aggregate_behavior=None) or aggregated scalar (if mean/max/min specified).

**Contract - Search Order:**
1. If `name` in `wrapper_states`: return that value (ignore aggregate_behavior)
2. Else if `name` in `optimizer.param_groups`:
   - `aggregate_behavior=None` → return list of values from all param groups
   - `aggregate_behavior="mean"` → return mean of values
   - `aggregate_behavior="max"` → return max of values
   - `aggregate_behavior="min"` → return min of values
3. Else: throw error (not found)

### _batch_received (Protected)

**For subclasses - required to use.**

Centralized counter management to maintain invariants consistently across all subclass implementations.

The _batch_received method handles centralized counter updates when a batch is processed. Call this at the start of your `step()` implementation, before any decision logic. The method increments `num_batches` (total batches processed since creation) and `num_draws` (batches accumulated since last optimizer step). It also enforces the `max_draws` safety bound - if calling this would push num_draws beyond max_draws, it throws an error to prevent unbounded accumulation.

Consult subclassing guide for details.

```python
def _batch_received(self) -> None
```

**Observable Effects**

- `num_batches` increases by one.
- `num_draws` increases by one.

### _take_optimizer_step (Protected)

**For subclasses - required to use.**

Call this in your `step()` implementation when your decision logic determines it's time to step the optimizer. This is the **only** way to step - directly calling `self.optimizer.step()` bypasses required processing and violates the contract.

```python
def _take_optimizer_step(self) -> None
```
Consult subclassing guide for details.

**Observable effects after calling:**
- `num_steps` increments by 1
- `num_draws` resets to 0
- `last_num_draws` set to previous `num_draws` value
- `last_grad_norm` updated with gradient information
- Base optimizer stepped (parameters updated)
- All gradients zeroed (ready for next accumulation)

**Throws:**
- `RuntimeError` if `num_draws == 0` (must accumulate at least one batch before stepping)

**Critical rule:**
Never call `self.optimizer.step()` directly. Always use `_take_optimizer_step()`. Bypassing this method breaks gradient accumulation and produces incorrect training.

### _bind_metric

A key function used to support distributed metrics. One specifies how to get the metric, and what to do if the system is replicated or sharded, as functions. The system then provides a .get_method call that is usable
The name of the metric, the function to read it with, the way to merge if replicated, and the way to merge if shared must all be provided.

```python
def _bind_metric(self,
                 name: str,
                 metric_reader: Callable[[Any], Numeric]),
                 replication_merger: Callable[[Numeric], Any],
                 sharded_merger: Callable[[Numeric], Any],
                 normal_merger: Callable[[Numeric], Any] = lambda x : x,
                ):
```

**For subclasses - required to use.**

**Parameters:**
- `name` (str) - The name this metric can be checked by
- `metric_reader` - A callable, usually bound to the system, that reads and processes the metric. It's input schema must be the same as you plan to invoke `._get_metric` with. 
- `replication_merger` - Handles 'replicated' distributed behavior, where the same model is in multiple locations. Exact behavior varies, but typically needs to take into account larger batch sizes or identicalness of model. Will be provided with the read metric. Return will pop out during ._get_metric
- `sharded_merger` - Handles 'sharded' distributed cases. This has the model placed on multiple devices instead. Has to account for having a single batch, and parameters distributed around. Will be provided with the result of metric_reader. Return will be returned by ._get_metric
- `normal_merger` - Active when no distributed mode is operational. Can be modified, but has a default as a passthrough.

**Contract**

- Sets up `._get_metric` to be invokable while handling distributed behavior.
- Can only handle replicated and sharded cases, not a mix.
### ._get_metric

**For subclasses - required to use.**

Gets a metric with all necessary distributed mode adjustments. This works by first invoking the stored `metric_reader' function, then passing it through normal_merger if no distributed state is set, replicated merger if replicated state is set, or sharded_merger if sharded mode is set. The metric reader function is passed in any arguments.

```python
def _get_metric(self, 
                name: str,
                *args,
                **kwargs
                )->Any:
```

**Parameters**

- **`name`**: The name of the metric to retrieve
- **`*args`**: A list of arguments to pass along
- **`**kwargs`**: A list of kwargs to pass along

**Return**:

Whatever ended up at the end of the factory. 


```python
def __aggregate_list(self, 
                     items: List[Union[torch.Tensor, float]],
                     aggregation: Literal["max", "min", "mean"],
                     )->float:
```


## Invariants

Properties that **must always hold:**

1. **`num_batches >= num_steps`** - Cannot step more than batches processed
2. **`num_draws <= max_draws`** - Cannot exceed maximum accumulation
3. **Gradient averaging** - Gradients always averaged before stepping
4. **One batch minimum** - At least one batch before any step
5. **State preservation** - `state_dict()` → `load_state_dict()` preserves exact state
6. **Transparent forwarding** - Non-overridden methods/attributes forward to `optimizer`
7. **Required utilities** - Subclasses must use `_batch_received()` and `_take_optimizer_step()`
8. **Namespace separation** - `_set_state()` cannot use optimizer param_group names
9. **Immutable flags** - Vital/optional status cannot change once set
11. **Starting state** - Counters are zero when initialized.
12. **

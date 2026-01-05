# Base object breakup

The base object is a monster. This is not unfortunately particularly maintainable. It, as of the time of this writing, has seven responsibilities loaded onto one object. This is to spec out the subobjects that will go into the main orchestrator to make the system modular and testable.

## Overview

The base object is broken up into the following objects, with the indicated responsibilities.

* **StateManagementSubsystem**: Location of all state-based manipulation and serialization. This includes managing wrapper state. Also sets up optimizers if needed, but does not do actual scheduling.
* **DistributedMetricsManagementSubsystem**: Location of metric bindings and the distributed metrics synchronization subsystem.
* **GradientAccumulationStepSubsystem**: Requires state management. Handles the counter tracking and gradient accumulation systems, exposes counter properties for forwarding, and a selection of public methods that are forwarded as private utilities for stepping.
* **ReportingSubsystem**: Creates statistics and vital statistic reports on demand. Requires StateManagementSubsystem, GradientAccumulationSubSystem.
* **OptimizerMockingMixin**: Mixin providing `__getattribute__` and `__setattr__` overrides to make wrapper transparently duck-type as wrapped optimizer while blocking direct state mutation after initialization.
* **OrchestratorMainSystem**: Main orchestrator, dependency injected with above subsystems, displays public API. Uses OptimizerMockingMixin. Deliberately does not directly construct for unit testing.
* **AbstractOptimizerWrapper**: Thin subclass of OrchestratorMainSystem that implements a constructor that automatically makes and resolves the dependencies. User Class

## Contracts

### StateManagementSubsystem

This system is responsible for everything to do with object state. Period. There is no setting of fields or mutating of public values that does not go through this system. It thus also serializes and unserializes

### Constructor

The constructor needs just one thing. an optimizer. 

### Fields

- **`wrapper_states`**: Dictionary for containing all wrapper information. Not externally usable.
- **`optimizer`**: Used for serialization and setting purposes.

### Methods

**get_state**

Retrieves state value by name from wrapper state or optimizer param groups.

```python
def get_state(self, name: str) -> Any:
```

1. If `name` in `wrapper_states`: return that value, without metadata
2. Else if `name` in `optimizer.param_groups`:
   - If value is not a scalar tensor or numeric type → throw
   - Return list of values from all param groups
3. Else: throw error (not found)


**set_state**

Sets up or overwrites state, with destination and priority rules. The only way to set 'fields' in the broader system. The optimizer set system should only be used during initialization to setup schedule binding sites. 

```python
def set_state(self,
               name: str,
               value: Any, 
               flag: Literal["vital", "optional", "optimizer"]
               ) -> None:
```

1. If flag is vital, optional, set value in wrapper_states with metadata "vital", "optional". Throw if setting name that exists to different kind of flag. 
2. If flag is optimizer
   - If name in optimizer throw with setting up optimizer field that already exists error.
   - If name not in optimizer use ScheduleAnything ExtendOptimizer to insert value into optimizer with name.

**show_state**

Reveals in terms of a list of tuples what keys exist, and what kind of key it is. On

```python
def show_state(self)->List[Tuple[str, str]]:
    """
    Lists all available state lookups in the class. 
    Tuples are first key name, and second one of "vital", "optional", and "optimizer
    """
```

1. Form list out of all entries in wrapper_states, with appropriate "vital" and "optional" marks.
2. For each entry shared across all parameter groups
   3. If the type is a scalar tensor or numeric datatype, include it as optimizer state.
4. Return list

**state_dict**

Form a working state dict that can be reloaded to resume training

```python
def state_dict(self) -> Dict[str, Any]:
```
1. Create a dictionary
2. Add "wrapper_states" entry into dictionary, consisting of wrapper states.
3. Add "optimizer_states" entry into dictionary, consisting of the results of optimizer.state_dict()
4. Return dictionary

**load_state_dict**

Reverses the above process. 

```python
def load_state_dict(self, state_dict: Dict[str, Any]) -> None
```

## Contracts

### DistributedMetricsManagementSubsystem

This subsystem is responsible for metric binding and metric resolution under distributed execution. All devices must agree on the same metrics for the system to work. This subsystem defines how named metrics are read and how their values are merged depending on the configured distributed execution mode. It does not mutate wrapper state, optimizer state, counters, or gradients.

This subsystem owns metric semantics and resolution policy. It does not own optimizer stepping, accumulation logic, reporting, or serialization.

### Invariants

1. Distributed execution affects metric resolution only.
2. Metric resolution is read-only.
3. Metric readers are executed exactly once per metric retrieval.
4. All registered metrics must have a complete resolution specification.
5. Distributed execution mode is fixed for the lifetime of the subsystem.

### Constructor

The constructor requires a distributed execution specification. The specification is validated eagerly and stored immutably.

```
def __init__(self, 
             distributed_state: Optional[Literal["replicated", "sharded"]]
             ):
```

**Rules:**

1. `distributed_state.distributed_mode` must be either `None`, `"replicated"`, or `"sharded"`.
2. Any other value causes construction to fail.
3. The distributed mode is treated as fixed and is not re-read dynamically.
4. The subsystem does not attempt to infer or detect distributed execution.

### Properties

- **distributed_mode**: The validated distributed execution mode. One of `None`, `"replicated"`, or `"sharded"`.

### Methods

**bind_metric**

Registers a metric and its resolution rules.

```python
def bind_metric(
    self,
    name: str,
    metric_reader: Callable[..., Any],
    replicated_merger: Callable[[Any], Any],
    sharded_merger: Callable[[Any], Any],
    normal_merger: Callable[[Any], Any],
) -> None:
```

1. `name` must not already be registered.
2. All callables must be provided and must be callable.
3. Registration is immutable; once a metric is bound, its definition cannot be replaced.
4. Binding records resolution policy only and performs no computation.

**get_metric**

Resolves a metric value according to the configured distributed execution mode.

```python
def get_metric(self, name: str, *args, **kwargs) -> Any:
```

1. Validate that `name` is registered.
2. Invoke the registered `metric_reader` with `*args` and `**kwargs`.
3. Select the resolution path based on `distributed_mode`.
4. Apply the corresponding merger.
5. Return the resolved value.

**Errors:**

- If `name` is not registered, throw with message indicating failure at **metric lookup**.
- If metric reader execution fails, throw with message indicating failure at **metric read**.
- If merging and distributed mode is `None` and error occurs throw specific error indicating origin from default context
- If  merging and distributed mode is `"replicated"` and error occurs throw specific error indicating origin from replicated context
- If merging and distributed mode is `"sharded"` and error occurs throw specific error indicating origin from sharded context.

### GradientAccumulationStepSubsystem

This subsystem manages gradient accumulation mechanics and optimizer stepping. It handles batch counting, enforces accumulation bounds, averages gradients before stepping, and maintains step counters.

This subsystem owns gradient accumulation and stepping mechanics. It does not own metric computation, state serialization, reporting, or distributed coordination.

**Dependencies:**
- Requires StateManagementSubsystem
- Requires optimizer reference

### Invariants

1. `num_batches >= num_steps` - Cannot step more than batches processed
2. `num_draws <= max_draws` - Enforced by batch_received
3. `num_draws == 0` immediately after stepping
4. `last_num_draws` and `last_grad_norm` are `None` before first step
5. Gradients are always averaged (divided by num_draws) before stepping

### Constructor

```python
def __init__(self,
             state_manager: StateManagementSubsystem,
             optimizer: torch.optim.Optimizer,
             max_draws: int = 64)
```

**Parameters:**
- `state_manager` - StateManagementSubsystem for persisting counters and cached values
- `optimizer` - PyTorch optimizer to wrap
- `max_draws` - Maximum batches allowed to accumulate before forced step

**Post-conditions:**

After construction, the following state exists:
- Vital state: `num_batches` = 0, `num_steps` = 0, `last_num_draws` = None, `last_grad_norm` = None
- Optional state: `num_draws` = 0
- Fields: `_state_manager`, `_optimizer`, `_max_draws`

### Fields

- `_state_manager`: StateManagementSubsystem reference
- `_optimizer`: torch.optim.Optimizer reference
- `_max_draws`: int configuration constant

### Properties

All properties retrieve values from state_manager.

- **num_batches** (`int`) - Total batches processed since creation
- **num_steps** (`int`) - Total optimizer steps taken
- **num_draws** (`int`) - Batches accumulated since last step
- **last_num_draws** (`Optional[int]`) - Batch count from most recent step. None before first step.
- **last_grad_norm** (`Optional[float]`) - Gradient norm from most recent step. None before first step.

### Methods

**batch_received**

Called when a batch is processed. Updates counters and enforces accumulation bounds.

```python
def batch_received(self) -> None:
```

1. Retrieve current num_draws from state_managerWht
2. If num_draws >= max_draws: raise RuntimeError indicating max accumulation exceeded
3. Increment num_batches by 1 via state_manager
4. Increment num_draws by 1 via state_manager

**take_optimizer_step**

Averages gradients, steps optimizer, updates counters. The only valid way to step the wrapped optimizer.

```python
def take_optimizer_step(self) -> None:
```

1. Retrieve current num_draws from state_manager
2. If num_draws == 0: raise RuntimeError indicating cannot step without batches
3. Average all gradients: divide each parameter's `.grad` by num_draws
4. Compute gradient norm using `compute_grad_norm_from_optimizer(optimizer)` utility
5. Step the wrapped optimizer
6. Zero all gradients on the wrapped optimizer
7. Update state via state_manager:
   - Set last_num_draws = current num_draws
   - Set last_grad_norm = computed norm
   - Increment num_steps by 1
   - Set num_draws = 0

### ReportingSubsystem

This subsystem generates statistics and vital statistics reports by querying StateManagementSubsystem. It is a stateless facade that discovers available state and formats it for reporting.

This subsystem owns statistics generation and formatting. It does not own state storage, computation, or mutation.

**Dependencies:**
- Requires StateManagementSubsystem

### Constructor

```python
def __init__(self, state_manager: StateManagementSubsystem)
```

**Parameters:**
- `state_manager` - StateManagementSubsystem for querying state

**Post-conditions:**

After construction, the following exists:
- Fields: `_state_manager`
- No internal state (stateless facade)

### Fields

- `_state_manager`: StateManagementSubsystem reference

### Methods

**aggregate_numeric_list**

Aggregates a list of numeric values using specified strategy.

```python
def aggregate_numeric_list(self,
                          values: List[Union[Number, torch.Tensor]],
                          behavior: Literal["mean", "max", "min"]) -> Number:
```

**Parameters:**
- `values` - List of numeric values (Number type from numbers module or scalar torch.Tensor)
- `behavior` - Aggregation strategy: "mean", "max", or "min"

**Returns:**
- Python numeric value (int or float), never torch.Tensor

**Tensor Handling:**
- Scalar tensors are converted to Python numbers via `.item()`
- Mixed lists of Numbers and tensors are supported
- Output is always a Python numeric type for JSON-serializability

1. Convert all values (tensors and Numbers) to Python numbers
2. If behavior is "mean": return mean of values
3. If behavior is "max": return max of values
4. If behavior is "min": return min of values

**statistics**

Generates complete or filtered statistics dictionary from available state. Implemented using aggregate_numeric_list() for lists. 

```python
def statistics(self,
               behavior: Literal["vital", "verbose"] = "verbose",
               aggregate_behavior: Literal["mean", "max", "min"] = "mean"
               ) -> Dict[str, Any]:
```

**Parameters:**
- `behavior` - "verbose" includes all state (vital, optional, optimizer). "vital" includes only vital and optimizer state.
- `aggregate_behavior` - How to aggregate multi-group hyperparameters or states when values differ

**Returns:**
- Dictionary with all values as Python native types (no torch.Tensors)

**Tensor Handling:**
- Scalar tensors are converted to Python numbers via `.item()`
- List entries containing tensors are converted element-wise
- Ensures all output values are JSON-serializable

1. Call `state_manager.show_state()` to get list of (name, flag) tuples
2. Filter based on behavior:
   - "verbose": include all entries
   - "vital": include only entries where flag is "vital" or "optimizer"
3. For each remaining entry:
   - Call `state_manager.get_state(name)` to retrieve value
   - If value is a list:
     - Check if all values in list are equal
     - If all equal:
       - Extract the scalar value
       - If scalar is a tensor, convert to Python number via `.item()`
       - Add to result dict with key name, value is the Python number or scalar
     - If not all equal:
       - Use `aggregate_numeric_list(value, aggregate_behavior)` to aggregate
       - (Note: aggregate_numeric_list internally converts any tensors to Python numbers)
       - Add to result dict with key name + "*" suffix (e.g., "lr*")
   - If value is a scalar tensor:
     - Convert to Python number via `.item()` and add to result dict
   - If value is a Number:
     - Add to result dict as-is
   - If value is a string AND flag is "vital" or "optional":
     - Add to result dict (allows metadata in wrapper state)
     - Optimizer params (flag="optimizer") cannot be strings - they must be numeric
   - All other types (None, objects, non-scalar tensors, etc.):
     - Omitted from result
   - Skip if retrieval or processing fails.
4. Return dictionary

**vital_statistics**

Generates curated vital statistics for real-time monitoring. Alias to statistics with behavior="vital".

```python
def vital_statistics(self, aggregate_behavior: Literal["mean", "max", "min"] = "mean") -> Dict[str, Any]:
```

**Parameters:**
- `aggregate_behavior` - How to aggregate multi-group optimizer parameters when values differ

1. Call and return `statistics(behavior="vital", aggregate_behavior=aggregate_behavior)`

### OptimizerMockingMixin

This mixin provides `__getattribute__` and `__setattr__` overrides to make the wrapper transparently duck-type as the wrapped optimizer. It allows the wrapper to satisfy `isinstance(obj, Optimizer)` checks while exposing the optimizer's interface, without inheriting Optimizer's abstract methods.

This mixin owns attribute forwarding mechanics only. It does not own state management, subsystem coordination, or optimizer operations.

**Dependencies:**
- Requires `_optimizer` field to exist
- Requires caller to invoke `_finalize_initialization()` at end of construction

### Fields

- `_initialized`: Boolean flag marking end of construction phase (set by `_finalize_initialization()`)

### Methods

**`_finalize_initialization`**

Marks end of construction phase. Must be called by orchestrator at end of `__init__`.

```python
def _finalize_initialization(self) -> None:
```

1. Set `_initialized = True` via `object.__setattr__`

**`__getattribute__`**

Forwards attribute access to wrapped optimizer while preserving wrapper's own interface.

```python
def __getattribute__(self, name: str) -> Any:
```

1. Walk MRO chain until reaching `Optimizer` class
2. For each class before `Optimizer`:
   - If name found in class `__dict__`: use normal object lookup and return
3. Check instance `__dict__` for name
4. If found in instance dict: return from instance
5. Else: retrieve `_optimizer` from instance dict and forward via `getattr(optimizer, name)`

**`__setattr__`**

Forwards attribute assignment to wrapped optimizer while allowing wrapper initialization and blocking post-init state mutation.

```python
def __setattr__(self, name: str, value: Any) -> None:
```

1. Retrieve instance `__dict__`
2. If `_initialized` not in instance dict:
   - Still in `__init__`, set locally via `object.__setattr__`
3. Else (after initialization):
   - Walk MRO until `Optimizer`, check if name in any class `__dict__`
   - If name in instance `__dict__` or found in wrapper class dicts: raise RuntimeError indicating collision with wrapper interface
   - Else: retrieve `_optimizer` from instance dict and forward via `setattr(optimizer, name, value)`

**Errors:**
- RuntimeError if attempting to set attribute that collides with wrapper's interface after initialization

### OrchestratorMainSystem

Main facade coordinating all subsystems and exposing unified public and protected API. Uses OptimizerMockingMixin for optimizer transparency. Dependency-injected with subsystems for testability.

This system owns the public interface and method coordination. It does not own state, metrics, accumulation, or reporting logic.

**Dependencies:**
- Requires StateManagementSubsystem
- Requires DistributedMetricsManagementSubsystem
- Requires GradientAccumulationStepSubsystem
- Requires ReportingSubsystem
- Requires optimizer reference
- Uses OptimizerMockingMixin

### Constructor

```python
def __init__(self,
             optimizer: torch.optim.Optimizer,
             state_manager: StateManagementSubsystem,
             distributed_metrics: DistributedMetricsManagementSubsystem,
             accumulation: GradientAccumulationStepSubsystem,
             reporting: ReportingSubsystem)
```

**Parameters:**
- `optimizer` - Wrapped PyTorch optimizer
- `state_manager` - StateManagementSubsystem instance
- `distributed_metrics` - DistributedMetricsManagementSubsystem instance
- `accumulation` - GradientAccumulationStepSubsystem instance
- `reporting` - ReportingSubsystem instance

**Algorithm:**

1. Store optimizer reference: `_optimizer = optimizer`
2. Store subsystem references: `_state_manager`, `_distributed_metrics`, `_accumulation`, `_reporting`
3. Call `self._finalize_initialization()` to complete construction

**Post-conditions:**
- All subsystem fields set
- `_initialized = True` (via OptimizerMockingMixin)
- Attribute access now forwards per mixin behavior

### Fields

- `_optimizer`: torch.optim.Optimizer reference
- `_state_manager`: StateManagementSubsystem reference
- `_distributed_metrics`: DistributedMetricsManagementSubsystem reference
- `_accumulation`: GradientAccumulationStepSubsystem reference
- `_reporting`: ReportingSubsystem reference

### Public Properties

**optimizer**

Direct access to wrapped optimizer.

```python
@property
def optimizer(self) -> torch.optim.Optimizer:
```

Returns the wrapped optimizer instance.

**num_batches**

Total batches processed since wrapper creation.

```python
@property
def num_batches(self) -> int:
```

Forwards to `_accumulation.num_batches`.

**num_steps**

Total optimizer steps taken.

```python
@property
def num_steps(self) -> int:
```

Forwards to `_accumulation.num_steps`.

**num_draws**1

Batches accumulated since last step.

```python
@property
def num_draws(self) -> int:
```

Forwards to `_accumulation.num_draws`.

**last_num_draws**

Number of batches in most recent optimizer step. None before first step.

```python
@property
def last_num_draws(self) -> Optional[int]:
```

Forwards to `_accumulation.last_num_draws`.

**last_grad_norm**

L2 gradient norm from most recent optimizer step. None before first step.

```python
@property
def last_grad_norm(self) -> Optional[float]:
```

Forwards to `_accumulation.last_grad_norm`.

**valid_schedule_targets**

List of all schedulable parameter names including optimizer native parameters and wrapper-extended parameters.

```python
@property
def valid_schedule_targets(self) -> List[str]:
```

1. Call `_state_manager.show_state()` to get list of (name, flag) tuples
2. Filter to entries where flag is "optimizer"
3. Extract and return list of names

**distributed_mode**

Configured distributed execution mode for metric aggregation.

```python
@property
def distributed_mode(self) -> Optional[Literal["replicated", "sharded"]]:
```

Forwards to `_distributed_metrics.distributed_mode`.

**device**

Device the optimizer's parameters are on. Returns device of first parameter.

```python
@property
def device(self) -> torch.device:
```

1. Access `_optimizer.param_groups[0]['params'][0]`
2. Return `.device` attribute

### Public Methods

**step**

Abstract method for subclasses to implement control algorithm. Called once per training batch.

```python
def step(self, *args, **kwargs) -> bool:
```

Subclasses must implement this to make step/accumulate decision. See base_object_api.md for subclassing contract.

**zero_grad**

Intentionally disabled. Always raises error.

```python
def zero_grad(self) -> None:
```

Raises NotImplementedError. Wrapper manages gradient zeroing internally.

**statistics**

Returns complete or filtered statistics dictionary for logging and debugging.

```python
def statistics(self,
               behavior: Literal["vital", "verbose"] = "verbose",
               aggregate_behavior: Literal["mean", "max", "min"] = "mean") -> Dict[str, Any]:
```

Forwards to `_reporting.statistics(behavior, aggregate_behavior)`. See ReportingSubsystem for details.

**vital_statistics**

Returns curated vital statistics for real-time monitoring (tqdm, dashboards).

```python
def vital_statistics(self, aggregate_behavior: Literal["mean", "max", "min"] = "mean") -> Dict[str, Any]:
```

Forwards to `_reporting.vital_statistics(aggregate_behavior)`.

**state_dict**

Serializes complete wrapper state for checkpointing.

```python
def state_dict(self) -> Dict[str, Any]:
```

Forwards to `_state_manager.state_dict()`.

**load_state_dict**

Restores wrapper state from checkpoint.

```python
def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
```

Forwards to `_state_manager.load_state_dict(state_dict)`.

### Protected Methods

For subclass implementation use only.

**_set_state**

Store wrapper state and expose parameters to ScheduleAnything.

```python
def _set_state(self, name: str, value: Any, flag: Literal["vital", "optional", "optimizer"]) -> None:
```

Forwards to `_state_manager.set_state(name, value, flag)`. See StateManagementSubsystem for details.

**_get_state**

Retrieve state from wrapper or optimizer through unified interface with optional aggregation.

```python
def _get_state(self,
               name: str,
               aggregate_behavior: Optional[Literal["mean", "max", "min"]] = None) -> Any:
```

1. Call `_state_manager.get_state(name)` to retrieve raw value
2. If `aggregate_behavior` is None: return value as-is
3. Else if value is a list: return `_reporting.aggregate_numeric_list(value, aggregate_behavior)`
4. Else: return value as-is (wrapper state, not a list)

**_bind_metric**

Register metric and its distributed resolution rules.

```python
def _bind_metric(self,
                 name: str,
                 metric_reader: Callable[..., Any],
                 replicated_merger: Callable[[Any], Any],
                 sharded_merger: Callable[[Any], Any],
                 normal_merger: Callable[[Any], Any] = lambda x : x,
) -> None:
```

Forwards to `_distributed_metrics.bind_metric(name, metric_reader, replicated_merger, sharded_merger, normal_merger)`. However, provides a default passthrough for normal merger.

**_get_metric**

Resolve metric value with distributed execution handling.

```python
def _get_metric(self, name: str, *args, **kwargs) -> Any:
```


Forwards to `_distributed_metrics.get_metric(name, *args, **kwargs)`.

**_batch_received**

Update counters when batch is processed. Call at start of step() implementation.

```python
def _batch_received(self) -> None:
```

Forwards to `_accumulation.batch_received()`.

**_take_optimizer_step**

Average gradients and step optimizer. Call when control algorithm decides to step.

```python
def _take_optimizer_step(self) -> None:
```

Forwards to `_accumulation.take_optimizer_step()`.

### AbstractOptimizerWrapper

User-facing class providing convenient constructor that automatically constructs and wires all subsystems. Subclasses implement step() for specific control algorithms.

This is the primary entry point for users and subclass implementers. It handles all subsystem construction and dependency wiring.

**Dependencies:**
- Subclass of OrchestratorMainSystem

### Constructor

Simplified constructor that auto-constructs all subsystems internally.

```python
def __init__(self,
             optimizer: torch.optim.Optimizer,
             max_draws: int = 64,
             distributed_mode: Optional[Literal["replicated", "sharded"]] = None)
```

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `max_draws` - Maximum batches that can accumulate before forcing step. Must be >= 1.
- `distributed_mode` - Distributed training mode: None (single device), "replicated" (DDP), or "sharded" (FSDP)

**Algorithm:**

1. Validate parameters:
   - optimizer is instance of torch.optim.Optimizer
   - max_draws >= 1
   - distributed_mode in [None, "replicated", "sharded"]
   - If distributed execution detected but distributed_mode is None: raise RuntimeError
2. Construct StateManagementSubsystem(optimizer)
3. Construct DistributedMetricsManagementSubsystem(distributed_mode)
4. Construct GradientAccumulationStepSubsystem(state_manager, optimizer, max_draws)
5. Construct ReportingSubsystem(state_manager)
6. Call super().__init__(optimizer, state_manager, distributed_metrics, accumulation, reporting)

**Errors:**
- TypeError if optimizer is not torch.optim.Optimizer
- ValueError if max_draws < 1
- ValueError if distributed_mode not in valid values
- RuntimeError if distributed execution detected but distributed_mode is None

**Post-conditions:**
- All subsystems constructed and wired
- OrchestratorMainSystem initialized
- Ready for subclass to bind metrics and initialize algorithm state via _bind_metric() and _set_state()

###

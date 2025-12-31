# Optimizer Wrapper API Reference

API reference for concrete optimizer wrapper implementations in Gradient Quality Control. For extending the base class, see [Base Object API](base_object_api.md). For usage patterns, see [User Guide](user_guide.md).

**Note:** Some wrappers are production algorithms (GNTS), while others are research controls for isolating specific effects (SBC, GNR, GNS).

## Navigation

- **[User Guide](user_guide.md)** - Usage patterns and library overview
- **[Base Object API](base_object_api.md)** - Abstract base class specification
- **[Wrapper Factories API Guide](api_guide.md)** - Convenience factory functions

---

## OptimizerWrapperSBC

**Scheduled Batch Controller** - Research control for fixed accumulation schedules.

### Constructor

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    physical_batch_size: int,
    max_batch_draws: int = 64
)
```

**Parameters:**
- `optimizer` - Configured PyTorch optimizer to wrap
- `physical_batch_size` - Size of each microbatch
- `max_batch_draws` - Maximum accumulation before forcing step (default: 64)

### Schedule Targets

- **`lr`** - Learning rate from wrapped optimizer
- **`weight_decay`** - Weight decay from wrapped optimizer (for Adam-family optimizers)
- **`logical_batch_size`** - Target total batch size injected by wrapper. Wrapper accumulates until reaching this size (rounded to nearest multiple of physical_batch_size).

### When to Use

Use when you want to fit arbitrary, possibly scheduled batch sizes in constant physical memory, or as a research baseline to isolate batch size effects from quality-based control decisions.

### Step

```python
def step(self, closure: Optional[Callable[[], Any]] = None) -> bool
```

**Returns:** True if optimizer stepped, False if still accumulating

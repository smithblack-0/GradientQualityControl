# Internal Utilities

Internal utility functions for GradientQualityControl. These are not part of the public API and are used by optimizer wrapper implementations.

---

## compute_gradient_norm

Computes the L2 norm of gradients across all parameters in an optimizer.

### Purpose

Provides a centralized, correct implementation of gradient norm computation used by multiple optimizer wrappers (GNTS, GNR, GNS). Ensures consistent gradient norm calculation across all control algorithms.

### Signature

```python
def compute_gradient_norm(optimizer: torch.optim.Optimizer) -> float
```

### Parameters

- `optimizer` (torch.optim.Optimizer) - Optimizer containing parameters with gradients

### Returns

- `float` - L2 norm of all gradients across all parameter groups

### Algorithm

1. Walk all parameter groups in the optimizer
2. Extract parameters from each group
3. Get gradient tensors from each parameter (`.grad`)
4. Pass gradient list to `torch.nn.utils.clip_grad_norm_()` with max_norm=inf to compute norm without clipping
5. Return the computed norm as float

### Contract

**Requires:**
- Optimizer must have at least one parameter group
- All parameters must have `.grad` populated (call after `.backward()`)

**Returns:**
- Combined L2 norm across all parameters: `sqrt(sum(grad.norm(2)^2 for grad in all_grads))`
- Returns 0.0 if no gradients present

**Critical:**
- Must NOT pass raw parameters to norm computation
- Must extract gradient tensors explicitly
- Uses PyTorch's `torch.nn.utils.clip_grad_norm_()` utility for correct norm calculation

### Usage Example

```python
# After backward pass
loss.backward()

# Compute gradient norm
grad_norm = compute_gradient_norm(optimizer)

# Use in control decision
if grad_norm <= threshold:
    optimizer._take_optimizer_step()
```

### Implementation Notes

This utility is internal to GradientQualityControl and not exposed in the public API. It exists to:
1. Prevent code duplication across GNTS, GNR, and GNS
2. Ensure gradient norm computation is always correct
3. Centralize the critical detail of extracting gradients (not parameters)

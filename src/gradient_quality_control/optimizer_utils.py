from numbers import Number
from typing import Any, Callable, List

import torch
from torch.optim import Optimizer


def optimizer_get_collection(
    optimizer: Optimizer,
    name: str,
) -> List[Any]:
    """
    Gets out of each param group a ordered list of the relevant elements.
    :param optimizer: The optimizer to get the
    :param name: The name. "params", "lr" are common. Extensions can be targetted too.
    :return: A list containing, in order, the targeted extracted feature from the param group
    """
    output = []
    for group in optimizer.param_groups:
        output.append(group[name])
    return output


def multiply_optimizer_gradients(optimizer: Optimizer, num: Number) -> None:
    """
    Multiplies all gradients in an optimizer by a number
    :param optimizer:Place to get gradients from
    :param num: Number to multiply by.
    """
    param_groups = optimizer_get_collection(optimizer, "params")
    for param_list in param_groups:
        for param in param_list:
            if param.grad is not None:
                param.grad *= num


def optimizer_get_raw_grad_norms(optimizer: Optimizer) -> List[float]:
    """
    Gets the grad norms for each discrete parameter group
    individually
    :param optimizer: Optimizer to get from
    :return: Unreduced groups.
    """
    param_groups = optimizer_get_collection(optimizer, "params")
    norms = []
    for param_list in param_groups:
        grads_cache = []
        for param in param_list:
            if param.grad is not None:
                grads_cache.append(param.grad)
        norms.append(torch.nn.utils.get_total_norm(grads_cache).item())
    return norms


def optimizer_get_grad_norm(optimizer: Optimizer) -> float:
    """
    Gets the gradient norm of the entire param group
    :param optimizer: The optimizer to get the gradient norm from
    :return: The result. Different groups are L2'd together.
    """
    raw_norms = optimizer_get_raw_grad_norms(optimizer)
    return torch.tensor(raw_norms).norm().item()


def compute_grad_norm_from_optimizer(optimizer: Optimizer) -> float:
    """
    Compute L2 norm of gradients across all parameters in optimizer.

    Provides centralized gradient norm computation used by gradient accumulation
    and control algorithms. Ensures consistent calculation across all wrappers.

    Args:
        optimizer: Optimizer containing parameters with gradients

    Returns:
        Combined L2 norm across all parameters

    Note:
        Returns 0.0 if no gradients are present.
        Must be called after .backward() to have populated gradients.
    """
    return optimizer_get_grad_norm(optimizer)


def setup_norm_logging_in_optimizer(optimizer: Optimizer) -> Callable[[], None]:
    """
    Attaches a callback hook that will cause parameters to
    end up with the norm of the last gradients that flowed
    through them on a field

    This is necessary as when performing gradient accumulation
    torch does not provide access to the gradient vectors before
    adding them into the accumulator unless intercepted

    A callable is returned that will release the hooks if invoked.
    """
    release = []
    parameters = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            parameters.append(p)

    for parameter in parameters:
        if hasattr(parameter, "_has_norm_logging"):
            continue

        def hook(grad: torch.Tensor, param=parameter):
            param._last_grad_norm = grad.norm()
            return grad

        parameter._has_norm_logging = True

        release.append(parameter.register_hook(hook))

    def release_hooks():
        for hook in release:
            hook.remove()

    return release_hooks


def get_last_grad_norm_from_optimizer(optimizer: Optimizer) -> float:
    """
    Gets the last grad norm that was logged.
    So long as you call this right after
    backwards you get the raw grad norms
    for the batch
    """
    norms = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            norms.append(p._last_grad_norm)
    norms = torch.tensor(norms)
    return norms.norm().item()

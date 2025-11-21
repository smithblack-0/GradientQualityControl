import math

from torch.optim.lr_scheduler import LambdaLR


def get_direct_cosine_annealing_with_warmup(
    optimizer,
    peak_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_value: float = 0.0,
) -> LambdaLR:
    """
    Create a scheduler with linear warmup and cosine annealing.
    Importantly, this DIRECTLY sets the value of the learning
    rate rather than multiplying it.

    Directly computes and sets values (via 'lr' param group):
    - Warmup: 0 → peak_value (linear)
    - Annealing: peak_value → min_value (cosine)

    Args:
        optimizer: Optimizer (wrapper) to schedule
        peak_value: Maximum value after warmup
        num_warmup_steps: Steps for warmup phase
        num_training_steps: Total training steps
        min_value: Minimum value at end (default: 0.0)

    Returns:
        LambdaLR scheduler that directly sets values
    """

    def lr_lambda(step):
        if step < num_warmup_steps:
            # Linear warmup from 0 to peak_value
            return peak_value * (step / num_warmup_steps)
        else:
            # Cosine annealing from peak_value to min_value
            progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_value + (peak_value - min_value) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_norm_threshold_cosine_annealing_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    start_norm: float = 1.0,
    end_norm: float = 0.0,
    warmup_multiplier: float = 10.0,
) -> LambdaLR:
    """
    Create a norm threshold scheduler with inverted warmup and cosine annealing.
    The norm threshold version has the questionable distinction that it's warmup
    is inverted, requiring instead starting with a threshold higher than the start
    norm target. This accomodates that, and directly sets the threshold after
    warmup to start norm then begins annealing.

    Schedule phases:
    - Warmup: (start_norm * warmup_multiplier) → start_norm (inverted, high to low)
    - Annealing: start_norm → end_norm (cosine)

    Args:
        optimizer: Optimizer (wrapper) to schedule
        start_norm: Norm threshold after warmup
        end_norm: Final norm threshold at end of training
        num_warmup_steps: Steps for warmup phase
        num_training_steps: Total training steps
        warmup_multiplier: Multiplier for warmup start (default: 10.0)

    Returns:
        LambdaLR scheduler for norm threshold values
    """
    warmup_start = start_norm * warmup_multiplier

    def lr_lambda(step):
        if step < num_warmup_steps:
            # Inverted warmup: high → start_norm
            progress = step / num_warmup_steps
            value = warmup_start - progress * (warmup_start - start_norm)
        else:
            # Cosine annealing: start_norm → end_norm
            progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            value = end_norm + (start_norm - end_norm) * cosine_decay
        return value

    return LambdaLR(optimizer, lr_lambda)


def get_quadratic_batch_schedule(
    optimizer,
    initial_batch_size: int,
    final_batch_size: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Batch size scheduler with quadratic growth from initial to final.

    Grows slowly early (small batches for exploration/noise),
    accelerates late (larger batches for efficiency).

    Args:
        optimizer: Optimizer (wrapper) to schedule
        initial_batch_size: Starting batch size
        final_batch_size: Ending batch size
        num_training_steps: Total training steps

    Returns:
        LambdaLR scheduler for batch size values
    """

    def lr_lambda(step):
        progress = step / num_training_steps
        # Quadratic growth: slow early, fast late
        return initial_batch_size + (final_batch_size - initial_batch_size) * (progress**2)

    return LambdaLR(optimizer, lr_lambda)


def get_curved_batch_schedule(
    optimizer,
    initial_batch_size: int,
    final_batch_size: int,
    num_training_steps: int,
    polynomial_exponent: float = 2.0,
) -> LambdaLR:
    """
    Batch size scheduler with arbitrary polynomial growth or decay from initial to final.

    Args:
        optimizer: Optimizer (wrapper) to schedule
        initial_batch_size: Starting batch size
        final_batch_size: Ending batch size
        num_training_steps: Total training steps
        polynomial_exponent:
            Literally what controls progress^polynomial_exponent.
            High valus have slow initial change followed shortly by
            more rapid change, low values are the opposite.
    Returns:
        LambdaLR scheduler for batch size values
    """
    assert polynomial_exponent > 0

    def lr_lambda(step):
        progress = step / num_training_steps
        # Quadratic growth: slow early, fast late
        return initial_batch_size + (final_batch_size - initial_batch_size) * (
            progress**polynomial_exponent
        )

    return LambdaLR(optimizer, lr_lambda)

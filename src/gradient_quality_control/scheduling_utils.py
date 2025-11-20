from torch.optim.lr_scheduler import LambdaLR


class NormWarmupScheduler(LambdaLR):
    """Scheduler with inverted warmup for norm targets."""

    def __init__(
        self,
        scheduler,
        num_warmup_steps,
        warmup_start=10.0,
        warmup_end=1.0,
    ):
        self.wrapped_scheduler = scheduler
        self.num_warmup_steps = num_warmup_steps
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

        def warmup_lr(step):
            progress = step / num_warmup_steps
            return warmup_start + progress * (warmup_end - warmup_start)

        super().__init__(scheduler.optimizer, warmup_lr, last_epoch=-1)

    def step(self):
        self.wrapped_scheduler.step()

        if self.last_epoch < self.num_warmup_steps - 1:
            super().step()

    def __getattr__(self, name):
        return getattr(self.wrapped_scheduler, name)


class NormWarmupAutoScheduler(LambdaLR):
    """Scheduler with inverted warmup for norm targets that auto-computes warmup range."""

    def __init__(
        self,
        scheduler,
        num_warmup_steps,
        warmup_start_multiplier=10.0,
    ):
        self.wrapped_scheduler = scheduler
        self.num_warmup_steps = num_warmup_steps
        self.warmup_start_multiplier = warmup_start_multiplier

        # Get the target (peak) lr from the wrapped scheduler's optimizer
        self.warmup_end = scheduler.optimizer.param_groups[0]["lr"]
        self.warmup_start = self.warmup_end * warmup_start_multiplier

        def warmup_lr(step):
            progress = step / num_warmup_steps
            return self.warmup_start + progress * (self.warmup_end - self.warmup_start)

        super().__init__(scheduler.optimizer, warmup_lr, last_epoch=-1)

    def step(self):
        self.wrapped_scheduler.step()

        if self.last_epoch < self.num_warmup_steps - 1:
            super().step()

    def __getattr__(self, name):
        return getattr(self.wrapped_scheduler, name)

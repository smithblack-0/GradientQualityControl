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

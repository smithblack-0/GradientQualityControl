# Gradient Quality Control User Guide

This guide is a comprehensive overview of the entire gradient quality control library's architecture, discussing the technical and mental models for use along with broad categories of implemented algorithms and a bit about the research background.

# Who this library is for?


This library is built to investigate and optimize a very specific niche. This niche is *automatic batch size tuning* using gradient accumulation while executing small scale *prototyping* and *lab* processes at production quality.

Out of this taxonomy falls two primary consequences

* **Research**: This library contains a variety of research objects including factories and optimizer wrappers which are either designed for this purpose or designed as controls for investigation.
* **Production Algorithms**: The main purpose is to find one of them. The best at the moment is GNTS (12/31/2025). These algorithms are best suited to prototyping, such as using DDP or single-thread training, or fine tuning.

Those interested in just a bit of tuning need only use the current production algorithm, which will have notes on distributed functionality on their factories and objects. 

# How to Use This Library

## The Big Idea


This library attempts to solve for the following objectives

* **Automatic batch size tuning:** Under all sorts of changes, and even some kinds of distributed work.
* **Easy to use**: Wrappers have good default factories, users only need to delve in as much complexity as they want.
* **Support for Replicated Distribution**: Using replicated distributed processes does not break anything.

These generally optimize towards small and medium research labs which perform rapid prototyping. 

## ScheduleAnything and Schedule Factories

The entire library is built on top of the ScheduleAnything library, which allows `generalized scheduling`. Thi is used to extend optimizers with additional features that can be bound to by schedules from ScheduleAnything. The core wrapper classes, which are subclasses of the `AbstractOptimizerWrapper` class, are injected with the optimizer to wrap, and then *pretend to be that optimizer*. However, secretly, under the hood they are deciding whether to accumulate more gradients or instead step. A key thing to keep in mind is `.zero_grad()` is no longer needed and no longer works.

This, in and of itself, is a completely capable description of the library, but it is not particularly intuitive. PyTorch users are not used to attaching multiple schedules in parallel. To accommodate that, each wrapper object comes with a set of factories for various common use cases that will construct both the wrapper optimizer and the schedule needed then return them. This makes using these objects through the factories largely a matter of passing the right parameters then hooking up the returns as normal in the training loop. 

For example:

```python
optimizer = ...
optimizer, schedule = make_optimizer_wrapper_with_trait(optimizer)
...
for batch in loader:
    loss = model(batch)
    loss.backwards()
    optimizer.step()
    schedule.step()
```

## The Sequential Binary Control Decision

The optimizer wrappers implement a process that formally lies within control theory called a 'Sequential Binary Decision Controller (SBDC)`. The wrappers, in reality, have remarkably little control over the training process. They, for the most part, make a single binary decision:

1) Invoke the wrapped .step() based on the observed metrics, get a mean gradient, step, and zero grads
2) Wait and accumulate more gradients.

This binary nature is where the formal control category comes from. Some minor deviations from this pattern occur, but they are explicitly noted in the documentation and mainly have to do with controls. This line of research is explicitly looking for the algorithm that fits best in this abstraction.

## The base class

The base class is responsible for handling accumulator management, statistics, state, and accumulation interffaces. It is located in [Base Object API](base_object_api.md). Subclasses are implemented primarily by implementing the .step function according to the contract, which corrolates directly with complying with the SBDC contract. 

Functions which users may wish to know about, and which all subclasses are contracted to provide, include

**Public Fields**:
- **`optimizer`** (`torch.optim.Optimizer`) - The wrapped optimizer, directly accessible

**Public Properties**:
- **`valid_schedule_targets`** (`List[str]`) - Read-only list of all schedulable parameter names. Includes native optimizer parameters (lr, weight_decay, momentum, etc.).

**Public Methods**
- **`step`**: Transparently duck-types exactly as before. No closure support.
- **`zero_grad`**: Now throws. The wrapper resets grads instead; do not use.
- **`statistics`**: Returns complete dictionary of features governing internal behavior.
- **`vital_statistics`**: Returns dictionary of things to display or log that are vital performance indicators; suitable for tqdm injection.
- **`state_dict`**: Gets the state dict, storing the wrapped optimizer and the state dict from the layer.
- **`load_state_dict`**: Losslessly resumes from a state dict.

## Optimizer Wrapper Summaries

The control optimizer wrappers which are available are enumerated as follows. ScheduleAnything binding targets are listed as well as general usage principles and motivation for the objects. It should be kept in mind going forward that when multiple batches are drawn before stepping, their gradients are meaned and thus on average get shorter. 

* **Scheduled Batch Controller**: A *control* controller for scheduling the logical batch size. It exposes for scheduling the entries "lr", "weight_decay", and "logical_batch_size". It is initialized with physical batch size, and will only invoke step once logical batch size from all steps to now exceed physical batch size. Generally, proper usage involves warming up the learning rate to a constant, warming up then cosine annealing the weight decay, and using some sort of inverse schedule, such as a low-to-high polynomial schedule, to start asking for a small number of batches that progressively increase.
* **Gradient Norm Rescaler**: A *control* wrapper which always steps, but also rescales the gradients to a constant scale during all steps before doing so. It exposes "weight_decay", "gradient_norm_target", and "lr" for scheduling. When used, it looks up the current gradient norm target, figures out the gradient norm, then rescales all gradients to that norm. It then immediately steps and zeros the gradients in the optimizer. Usage is best as learning rate to constant warmup, weight decay to warmup then cosine annealing, and cosine annealing of norm, as it replaces learning rate scheduling by directly controlling the length of the gradient instead. Used to isolate how much gain comes from more consistent gradient lengths.

Optimizer wrappers which are or were intended to increase performance rather than isolate ability are

* **Metric Hypothesis Test**: A controller that makes it's step decision based on variance in an observed metric. Once, given the metric samples, the controller is %confidence_level sure that the true metric has been found such that the confidence interval is under %percent_error. Features "lr", "confidence_level", and "percent_error_threshold" are exposed to ScheduleAnything. A typical use case would have one scheduling the learning rate using a cosine schedule, and confidence level and percent error threshold both warmup to a constant to allow early steps to proceed rapidly. The typical metric is the loss per drawn batch.
* **Gradient Noise Scale**: Largely a failed mechanism that did not correctly optimize performance. It is based on the GNS metric from traditional optimizer theory, but did not corrolate well with the real world. Instead, it now exposes "lr" and "noise_tolerance"; noise tolerance is literally what the noise to signa ratio we are willing to tolerate. This never performed well under our optimization processes. It delays taking steps until the GNS is below a threshold weighted by the cost of processing one more batch.
* **Gradient Norm Threshold Scheduling**: The current best algorithm. This works by associating the length of the gradients to be a proxy to gradient quality, and demanding the gradients be under a certain length before taking a step, allowing cancellation. The features "lr", "weight_decay", and "gradient_norm_threshold" are primary schedule targets. Normal operation is to warmup learning rate to a constant, warmup weight decay then cosine anneal, and inverse warmup then cosine anneal the gradient norm threshold. Like the Gradient Norm Rescaler this directly controls the length of the gradients, eliminating the need for a learning rate schedule.

For more details on the actual objects, consult [Optimizer Wrapper API](optimizer_wrapper_api.md).

## Wrapper Factories

The set of wrapper factories exist to make it a bit easier to bind up the optimizer wrappers to the need schedules or schedule possibilities in a convenient to use package. Some of these are production-ready algorithms as well. The set of wrapper factories are

* make_sbc_with_polynomial_schedule: Allows following a polynomial curve from initial batch size to final batch size with an included warmup; usable to schedule the batch size directly instead. Learning rate is to constant, batch size is by polynomial schedule, and weight decay is by cosine annealing to account for removed learninig rate schedule.
* make_gns_with_cosine_annealing_schedule: Allows the scheduling of the gradient noise scale response quality with cosine annealing and thus adjusted the batch size. Learning rate is by cosine annealing, and the noise scale also follows such a schedule
* make_gns_default: The default schedule. This inverse warms up the noise tolerance then it just sticks there; learning rate follows a cosine schedule. 
* make_gnr_with_cosine_annealing_schedule_and_lr_to_constant: The gradient norm rescaler class just rescales the gradient norms to be a certain size then immediately steps. This implements a schedule that is bound to that. Learning rate warms up to constant.
* * make_gnr_with_cosine_annealing_schedule_and_lr_cosine_anneals: The gradient norm rescaler class just rescales the gradient norms to be a certain size then immediately steps. This implements a schedule that is bound to that. Learning rate anneals too according to the same schedule
* make_gnts_with_cosine_annealing_schedule: Learning rate warms up to constant, threshold inverse warmup to starting value, then cosine annealing to ending value. Weight decay warms up to completely on, then cosine anneals to zero over timesteps.

For more details consult [Wrapper Factories API Guide](api_guide.md).


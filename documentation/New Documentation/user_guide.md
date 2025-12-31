# Gradient Quality Control User Guide

This guide is a comprehensive overview of the entire gradient quality control library's architecture, discussing the technical and mental model for use along with broad categories of implemented algorithms

# Who this library is for?


This library has a very specific niche: *Improve the quality of gradients before optimizers take a step*. Under this taxonomomy, a variety of research and production tools have been implemented. This includes research controls, but, and more relevantly for most, a variety of practical algorithms providing convenient methods of using some of these algorithm.

This library is a **research tool** for people who understand optimization and need precise control over gradient quality. It is also a **production component** for those who want to hook up and use the flagship algorithm. While the flagship algorithm will likely be the most useful, it also has other algorithms to act as research controls or options. 


# How to Use This Library

## Usage Overview

The primary interface of the library is the optimizer wrapper. Specifically, the `AbstractOptimizerWrapper` class defined in the base.py file (see [Base Object API](base_object_api.md) for full specification). This is an abstract optimizer designed to be injected with a real optimizer and which then makes decisions on when to invoke the `.step()` and `.zero_grad()` functionality, taking this over from the user. Under the hood, this may implement gradient changes and operations such as rescaling the gradients, but it is most often implemented using functionality to perform gradient accumulation until finally taking a step, at which point the optimizer steps in the direction of the average gradient.

Each concrete implementation of the `AbstractOptimizerInterface`. such as `OptimizerWrapperGNTS`, are paired with one or more factory methods called `make_#variety_#details' which returns a wrapped optimizer and a bound schedule configuration believed to be useful. These are then used like normal as part of a training schedule:

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

The only thing these controllers have any control over at a physical level is

1) Do we step the optimizer then zero the grads?
2) Do we fiddle with the gradients at all before taking that step?

They may sometimes receive external inputs, such as metrics. During the following discussion, we will presume we are binding to an optimizer with the following fields:

* 'lr': The learning rate for an AdamW process
* 'weight_decay': The weight decay for an AdamW process
* All other relevent optimizer fields


## Optimizer Wrapper Summaries

The control optimizer wrappers which are available are enumerated as follows. ScheduleAnything binding targets are listed as well as general usage principles and motivation for the objects. It should be kept in mind going forward that when multiple batches are drawn before stepping, their gradients are meaned and thus on average get shorter.

* **Scheduled Batch Controller**: A *control* controller for scheduling the logical batch size. It exposes for scheduling the entries "lr", "weight_decay", and "logical_batch_size". It is initialized with physical batch size, and will only invoke step once logical batch size from all steps to now exceed physical batch size. Generally, proper usage involves warming up the learning rate to a constant, warming up then cosine annealing the weight decay, and using some sort of inverse schedule, such as a low-to-high polynomial schedule, to start asking for a small number of batches that progressively increase.
* **Gradient Norm Rescaler**: A *control* wrapper which always steps, but also rescales the gradients to a constant scale during all steps before doing so. It exposes "weight_decay", "gradient_norm_target", and "lr" for scheduling. When used, it looks up the current gradient norm target, figures out the gradient norm, then rescales all gradients to that norm. It then immediately steps and zeros the gradients in the optimizer. Usage is best as learning rate to constant warmup, weight decay to warmup then cosine annealing, and cosine annealing of norm, as it replaces learning rate scheduling by directly controlling the length of the gradient instead. Used to isolate how much gain comes from more consistent gradient lengths.

Optimizer wrappers which are or were intended to increase performance rather than isolate ability are

* **Metric Hypothesis Test**: A controller that makes it's step decision based on variance in an observed metric. Once, given the metric samples, the controller is %confidence_level sure that the true metric has been found such that the confidence interval is under %percent_error. Features "lr", "confidence_level", and "percent_error_threshold" are exposed to ScheduleAnything. A typical use case would have one scheduling the learning rate using a cosine schedule, and confidence level and percent error threshold both warmup to a constant to allow early steps to proceed rapidly. The typical metric is the loss per drawn batch.
* **Gradient Noise Scale**: Largely a failed mechanism that did not correctly optimize performance. It is based on the GNS metric from traditional optimizer theory, but did not corrolate well with the real world. Instead, it now exposes "lr" and "noise_tolerance"; noise tolerance is literally what the noise to signa ratio we are willing to tolerate. This never performed well under our optimization processes. It delays taking steps until the GNS is below a threshold weighted by the cost of processing one more batch.
* **Gradient Norm Threshold Scheduling**: The current best algorithm. This works by associating the length of the gradients to be a proxy to gradient quality, and demanding the gradients be under a certain length before taking a step, allowing cancellation. The features "lr", "weight_decay", and "gradient_norm_threshold" are primary schedule targets. Normal operation is to warmup learning rate to a constant, warmup weight decay then cosine anneal, and inverse warmup then cosine anneal the gradient norm threshold. Like the Gradient Norm Rescaler this directly controls the length of the gradients, eliminating the need for a learning rate schedule.


For more details on the actual objects, consult [Base Object API](base_object_api.md), or for details on why these are the right way to think about the abstractions consult [Research Guide](research_guide.md).

## Wrapper Factories

The set of wrapper factories exist to make it a bit easier to bind up the optimizer wrappers to the need schedules or schedule possibilities in a convenient to use package. Some of these are production-ready algorithms as well. The set of wrapper factories are

* make_sbc_with_polynomial_schedule: Allows following a polynomial curve from initial batch size to final batch size with an included warmup; usable to schedule the batch size directly instead. Learning rate is to constant, batch size is by polynomial schedule, and weight decay is by cosine annealing to account for removed learninig rate schedule.
* make_gns_with_cosine_annealing_schedule: Allows the scheduling of the gradient noise scale response quality with cosine annealing and thus adjusted the batch size. Learning rate is by cosine annealing, and the noise scale also follows such a schedule
* make_gns_default: The default schedule. This inverse warms up the noise tolerance then it just sticks there; learning rate follows a cosine schedule. 
* make_gnr_with_cosine_annealing_schedule_and_lr_to_constant: The gradient norm rescaler class just rescales the gradient norms to be a certain size then immediately steps. This implements a schedule that is bound to that. Learning rate warms up to constant.
* * make_gnr_with_cosine_annealing_schedule_and_lr_cosine_anneals: The gradient norm rescaler class just rescales the gradient norms to be a certain size then immediately steps. This implements a schedule that is bound to that. Learning rate anneals too according to the same schedule
* make_gnts_with_cosine_annealing_schedule: Learning rate warms up to constant, threshold inverse warmup to starting value, then cosine annealing to ending value. Weight decay warms up to completely on, then cosine anneals to zero over timesteps.

For more details consult [Wrapper Factories API Guide](api_guide.md).


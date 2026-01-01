# Gradient Quality Control User Guide

This guide is a comprehensive overview of the entire gradient quality control library's architecture, discussing the technical and mental models for use along with broad categories of implemented algorithms and a bit about the research background.

# Who is this library for?


This library is built to investigate and optimize a very specific niche. This niche is *automatic batch size tuning* using gradient accumulation while executing small scale *prototyping* and *lab* processes at production quality.

Out of this taxonomy fall two primary consequences

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

The entire library is built on top of the ScheduleAnything library, which allows `generalized scheduling`. This is used to extend optimizers with additional features that can be bound to by schedules from ScheduleAnything. The core wrapper classes, which are subclasses of the `AbstractOptimizerWrapper` class, are injected with the optimizer to wrap, and then *pretend to be that optimizer*. However, secretly, under the hood they are deciding whether to accumulate more gradients or instead step. A key thing to keep in mind is `.zero_grad()` is no longer needed and no longer works.

This, in and of itself, is a completely capable description of the library, but it is not particularly intuitive. PyTorch users are not used to attaching multiple schedules in parallel. To accommodate that, each wrapper object comes with a set of factories for various common use cases that will construct both the wrapper optimizer and the schedule needed then return them. This makes using these objects through the factories largely a matter of passing the right parameters then hooking up the returns as normal in the training loop. 

For example:

```python
optimizer = ...
optimizer, schedule = make_optimizer_wrapper_with_trait(optimizer)
...
for batch in loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    schedule.step()
```

## The Sequential Binary Control Decision

The optimizer wrappers implement a process that formally lies within control theory called a 'Sequential Binary Decision Controller (SBDC)`. The wrappers make a single binary decision:

1) Invoke the wrapped .step() based on the observed metrics, get a mean gradient, step, and zero grads
2) Wait and accumulate more gradients.

This binary nature is where the formal control category comes from. Some minor deviations from this pattern occur, but they are explicitly noted in the documentation and mainly have to do with controls. This line of research is explicitly looking for the algorithm that fits best in this abstraction.

## The base class

The base class is responsible for handling accumulator management, statistics, state, and accumulation interffaces. It is located in [Base Object API](base_object_api.md). Subclasses are implemented primarily by implementing the .step function according to the contract, which correlates directly with complying with the SBDC contract. 

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

The control optimizer wrappers which are available are enumerated as follows. ScheduleAnything binding targets are listed as well as general usage principles and motivation for the objects. It should be kept in mind going forward that when multiple batches are drawn before stepping, their gradients are meaned and thus on average get shorter. *Note that using these objects directly requires attaching ScheduleAnything schedules.* As such, it is usually recommended to use the factories in the following section unless you need custom behavior.

| Full Name (abreviation)          |   Object Name      |   Link   | Purpose                                                                      |
|----------------------------------|--------------------|----------|------------------------------------------------------------------------------|
| Scheduled Batch Controller (SBC) | OptimizerWrapperSBC|[SBC](optimizer_wrapper_api.md#OptimizerWrapperSBC)| Schedule a logical batch size in constant physical batch size                |
| Gradient Norm Rescaler (GNR)     | OptimizerWrapperGNR| [GNR](optimizer_wrapper_api.md#OptimizerWrapperGNR)| Always step; Rescale gradient norm to scheduled length                       |
|Metric Hypothesis Test (MHT)      | OptimizerWrapperMHT| [MHT](optimizer_wrapper_api.md#OptimizerWrapperMHT)| Adaptively vary batch size by sensing variance in the loss, or other metrics |
|Gradient Noise Scale (GNS)| OptimizerWrapperGNS| [GNS](optimizer_wrapper_api.md#OptimizerWrapperGNS)| A control based on McCandish's work|
| **Gradient Norm Threshold Scheduler** | OptimizerWrapperGNTS | [GNTS](optimizer_wrapper_api.md#OptimizerWrapperGNTS)| Accumulate gradients until gradient norm is below threshold, then step. Best algorithm|

For more details on the actual objects, consult [Optimizer Wrapper API](optimizer_wrapper_api.md) or follow the appropriate link.

## Scheduling Factories

Using the wrappers directly requires attaching generalized schedules, usually through ScheduleAnything. However, each varient of wrapper has researched default factories available as well. These factories will set up the schedules as well, making them considerably easier for the lay user to use, but make some assumptions about optimal schedule behavior in exchange. Anything marked as "conventional_lr" schedules learning rate with cosine annealing. Non-conventional variants warmup learning rate to constant instead. All factories expect optimizers with `lr` and `weight_decay` parameters (like AdamW). Parameters not present are skipped with a warning.

| Name                                                     | Link                                                                          | Purpose                                                                                                         |
|----------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| make_sbc_with_polynomial_schedule                        | [link](api_guide.md#make_sbc_with_polynomial_schedule)                        | Schedule the batch size instead of the learning rate                                                            |
| make_sbc_with_polynomial_schedule_conventional_lr        | [link](api_guide.md#make_sbc_with_polynomial_schedule_conventional_lr)        | Like last one, but includes a cosine annealing learning rate schedule and no weight decay scheduling.           |
| **make_gnts_with_cosine_annealing_schedule**             | [link](api_guide.md#make_gnts_with_cosine_annealing_schedule)                 | Adaptive and reactive batch resizing using gradient norms; best algorithm                                       |
| make_gnts_with_cosine_annealing_schedule_conventional_lr | [link](api_guide.md#make_gnts_with_cosine_annealing_schedule_conventional_lr) | Like the primary algorithm, except we retain cosine annealing of learning rate and do not schedule weight decay |
|make_gnr_with_cosine_annealing_schedule| [link](api_guide.md#make_gnr_with_cosine_annealing_schedule)                  | Rescales the gradients to match a target, then cosine anneals that target|
|make_gnr_with_cosine_annealing_schedule_conventional_lr| [link](api_guide.md#make_gnr_with_cosine_annealing_schedule_conventional_lr)  | Eliminates the weight decay scheduling in favor of a conventional cosine annealing learning rate schedule|
|make_gns_with_cosine_annealing_schedule| [link](api_guide.md#make_gns_with_cosine_annealing_schedule)| Annealed threshold for better performance of the gradient noise scale|
|make_gns_default| [link](api_guide.md#make_gns_default)| Warmup to threshold, then just runs.|
|make_mht_with_warmup_schedule| [link](api_guide.md#make_mht_with_warmup_schedule)| Adaptively control batch size by detecting loss variance|

For more details consult [Wrapper Factories API Guide](api_guide.md).


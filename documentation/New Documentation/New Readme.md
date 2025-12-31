# Gradient Quality Control

**Gradient Quality Control (GQC)** is a research/infrastructure library designed to examine a wide variety of 'gradient preprocessers' and deliver the current best production solutions that significantly improve training quality of a large variety of small and medium sized models.

It provides a well-defined abstract class to do research through, drop-in optimizer wrappers built on top of the robust ScheduleAnything package, and a variety of gradient cleaners to experiment with, some of which are quite powerful. It also has a flagship implementation that is kept up to date for intermediate rather than advanced users.

## Why Gradient Quality Control?

An enormous amount of effort has gone into examining optimizer theory in machine learning, with options like Adam, AdamW, AdaGrad, etc. Surprisingly little research, however, has gone into deciding when to take an optimizer step in the first place or how many samples to draw, with AdaBatch being one notable example. Even less research has observed you can control the logical batch size in constant memory by using gradient accumulation and defining when to step as a control decision. And nothing, as far as we can tell, has observed that how big the batch is can be composed orthogonally to standard optimizer theory using gradient accumulation and a .step()/continue_to_accumulate sequential binary control decision.

This library fits in that missing niche. We define an abstract optimizer wrapper class for gradient accumulation fitting formally in the Sequential Binary Decision Controller control theory niche; concrete cases then implement the control algorithm. Decision constants like thresholds or targets are defined as schedulable parameters using ScheduleAnything. Some algorithms have shown incredible promise at this point, and the library is moving towards production-ready usage with some flagship algorithms.

The line of research have been designed specifically with assisting small-scale labs and prototyping in mind. The library specifically includes algorithms that seek to maintain the best balance of the following.

1) Maintains largely the same logical batch size regardless of the underlying physical batch size. 
2) Seeks out and maintains, as best as possible, the best logical batch size tuning when the same model is scaled up or down in size.
3) Seeks out and maintains, as best as possible, the best logical batch size tuning regardless of the underlying dataset.
4) Support this with defaults that largely 'just work' for common LLM pretraining activities.
5) React to low quality data and increase the batch size to compensate.

In other words, **this line of research is explicitly to remove a hyperparameter tuning pain point in prototyping and research.** If batch size tuning is easy, it means small labs iterate much quicker. You also get a bit of extra performance on top, for free, and this effect should persist to scale, though this has not yet been verified.

## Who needs this?

This library is designed exclusively for the PyTorch ecosystem, and has been tried primarily as an LLM pretraining aid, though it may end up useful for fine tuning too. The exact

### Practitioners

If you are doing one of the following things.

* Prototyping in under 10B parameters on a single device or with PyTorch DDP and do not want to have to fine tune your batch size.
* Looking for a solution that does not require retuning every time you change model sizes or physical batch size.

Then the flagship algorithm shown in the Getting Started guide is likely an excellent choice. It will ensure your batch size is approximately ideal, letting you focus on prototyping rather than worrying if your prototype is not useful, or the batch size was just set correctly. Other than that, it will largely stay out of your way. If you are:

* Moving towards a large scale production capability.
* Intending to use FSDP.
* Going to use non-replicated distributed functionality.

Then you are currently unsupported and would have to implement your own algorithm. Practioners should consult the [User Guide](user_guide.md) then the [Wrapper Factories API Guide](api_guide.md) if needed. Users performing advanced actions may consult the [Base Object API](base_object_api.md) as well.

### Researchers

If you are a researcher interested in this line of reseearch, it is recommended to consult first the [Research Guide](research_guide.md) which will provide commentary about the entire line of research, then the [User Guide](user_guide.md) which will discuss how the library is implemented, and [Base Object API](base_object_api.md) for underlying details of the algorithms.

## Getting Started 

This algorithm is continously maintained as the example of the most productive implementation yet discovered. It has last been updated on 12/28/2025

Getting started with GQC is straightforward. 

First, install the library from PyPi

```text
pip install torch-gqc
```

Now, suppose we have a classical learning loop, something like

```python
train_loader = get_train_loader(batch_size = 64)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_cosine_annealing_schedule(optimizer, warmup_steps=500, ...)
for inputs, labels in train_loader:
    
    # Loss
    logits = model(inputs)
    loss = cross_entropy(logits, labels)
    loss.backward()
    
    # Optimization
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
```

In the Gradient Norm Theshold Scheduling algorithm, instead, we would directly control the step size and signal-to-noise ratio by demanding the gradient norm be a certain magnitude before stepping. This involves several manipulations under the hood and is implemented as a function returning an optimizer wrapper and a schedule. For this reason, this is composable with basically any existing optimizer. The defaults have been tuned on AdamW and GPT2.

```python
from gradient_quality_control import OptimizerWrapperGNTS, make_gnts_with_cosine_annealing_schedule

...
train_loader = get_train_loader(batch_size = 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer, schedule = make_gnts_with_cosine_annealing_schedule(optimizer,
                                                             num_warmup_steps = 500,
                                                             num_training_steps = 10000
                                                             )
for inputs, labels in train_loader:
    
    # Loss
    logits = model(inputs)
    loss = cross_entropy(logits, labels)
    loss.backward()
    
    # Optimization. IMPORTANT! No zero grad anymore, optimizer now takes care of that.
    optimizer.step()
    schedule.step()
```

The primary control available is to set the norm schedule, specifically what it ends up warming up into and annealing down to. This literally specifies to the system how low the gradient norms have to become before the system can take an optimizer step. It should be kept in mind **lower numbers are more restrictive**. If the system never steps during warmup, you can also set initial_warmup_multiplier to a higher value.

```python
from gradient_quality_control import OptimizerWrapperGNTS, get_gnts_optimizer_and_cosine_schedule

...
train_loader = get_train_loader(batch_size = 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer, schedule = get_gnts_optimizer_and_cosine_schedule(optimizer, 
                                                             num_warmup_steps = 500,
                                                             num_training_steps = 10000,
                                                             norm_warmup_target=0.95, # Default value
                                                             norm_anneal_target=0.25, # Default vlaue
                                                             initial_warmup_multiplier = 10.0 # Default value.
                                                             )
for inputs, labels in train_loader:
    
    # Loss
    logits = model(inputs)
    loss = cross_entropy(logits, labels)
    loss.backward()
    
    # Optimization. IMPORTANT! No zero grad anymore, optimizer now takes care of that.
    optimizer.step()
    schedule.step()
```

A wide variety of control statistics are also available by means of the statistics method, and a set of well-chosen vital statistics relevant to the optimizer and suitable for tqdm display or logging are available through .vital_statistics()

```python
from gradient_quality_control import OptimizerWrapperGNTS, get_gnts_optimizer_and_cosine_schedule

...
train_loader = get_train_loader(batch_size = 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer, schedule = get_gnts_optimizer_and_cosine_schedule(optimizer, 
                                                             num_warmup_steps = 500,
                                                             num_training_steps = 10000,
                                                             norm_warmup_target=0.95,
                                                             norm_anneal_target=0.25,
                                                             )

pbar = tqdm(train_loader, desc="Training")

for inputs, labels in train_loader:
    
    # Loss
    logits = model(inputs)
    loss = cross_entropy(logits, labels)
    loss.backward()
    
    # Optimization. IMPORTANT! No zero grad anymore, optimizer now takes care of that.
    optimizer.step()
    schedule.step()
    vital_statistics = optimizer.get_vital_statistics()
    pbar.set_postfix(vital_statistics)
```

Consider consulting the production guide if needed, or the api guide, if more information on how to control the system is requested. Overall, however, you can think as the norm threshold as a quality demand. Lower threshold requests higher quality gradients.

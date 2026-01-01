# Gradient Quality Control

**Gradient Quality Control (GQC)** is a research/infrastructure library designed to examine a wide variety of 'gradient preprocessers' which improve performance before an optimizer ever gets the chance to step.

It provides  drop-in optimizer wrappers built on top of the robust ScheduleAnything package, a well-defined abstract class to do research through, and a variety of wrapper factories to experiment with, some of which are quite powerful. It also has a flagship implementation that is kept up to date for intermediate rather than advanced users.

## Documentation

- **[User Guide](documentation/user_guide.md)** - Comprehensive overview of library architecture and usage patterns
- **[Wrapper Factories API Guide](documentation/api_guide.md)** - Pre-configured factory functions for common use cases
- **[Optimizer Wrapper API](documentation/optimizer_wrapper_api.md)** - Detailed specifications for individual wrapper algorithms
- **[Base Object API](documentation/base_object_api.md)** - Abstract base class for extending with custom algorithms
- **[Research Guide](documentation/research_guide.md)** - Commentary on the research line and design decisions

## What is Gradient Quality Control?

An enormous amount of effort has gone into examining optimizer theory in machine learning, with options like Adam, AdamW, AdaGrad, etc. Surprisingly little research, however, has gone into deciding when to take an optimizer step in the first place, with AdaBatch being one notable example. Even less research has used gradient accumulation to dynamically control batch size. And nothing, as far as we can tell, has observed that how big the batch is can be composed orthogonally to standard optimizer theory using gradient accumulation and a .step()/continue_to_accumulate sequential binary control decision.

This library fits in that missing niche. We define an abstract optimizer wrapper class for gradient accumulation fitting formally in the Sequential Binary Decision Controller control theory niche; concrete cases then implement the control algorithm. Decision constants like thresholds or targets are defined as schedulable parameters using ScheduleAnything. Some algorithms have shown incredible promise at this point, and the library is moving towards production-ready usage with some flagship algorithms.

## Why would I want it?

Four main reasons.

1) **The flagship algorithm reduces hyperparameter tuning**: The GNTS algorithm is not tuning-free under all circumstances, but several orders of magnitude tuning lighter. You may vary physical batch size, dataset, model size, and other features across several orders of magnitude before performance degrades below an acceptable level, and the system is robust to minor architecture changes. Tuning once for small, medium, and large model sizes then varying arbitrarily within the size is possible. 
2) **It gains you a little extra performance on well-tuned models**: Even when the model has already been tuned, the reactive nature of the GNTS controller ensures it draws more batches when the gradients gets noisy, which tends to improve training efficiency by 10-20%. 
3) **It works out of the box in most distributed environments**: The library will tell you when it needs additional distributed information, and naturally supports distribution on all algorithms. 
4) **It naturally adapts to noise in exploratory research**: Since GNTS is reactive to noise in the gradients, it should allow much faster iteration during exploratory research in novel models which may implement layers that increase the noise level shifting where the ideal batch size is. This makes it significantly easier to avoid false negatives when you do not have funding for hyperparameter tuning.

Overall, the most exciting consequence may be the democratization effect. It is possible for small groups to do research or training that used to require a much larger budget, while not significantly increasing the capacities at the frontier end of the scale, increasing research churn and efficiency. This was, in fact, the primary design goal of this line of research.

### Practitioners

If you are doing one of the following things.

* Prototyping in under 10B parameters on a single device or with PyTorch DDP or FSDP and do not want to have to fine tune your batch size.
* Looking for a solution that does not require retuning every time you change model sizes or physical batch size.

Then the flagship algorithm shown in the Getting Started guide is likely an excellent choice. It will ensure your batch size is approximately ideal, letting you focus on prototyping rather than worrying if your prototype is not useful, or the batch size was just set correctly. Other than that, it will largely stay out of your way. If you are:

* Moving towards a large scale production capability.
* Going to use capacity that has some replication of parameters and some sharding.

Then you are currently unsupported and would have to implement your own algorithm or use multiple optimizers. Practitioners should consult the [User Guide](documentation/user_guide.md) then the [Wrapper Factories API Guide](documentation/api_guide.md) if needed. Users performing advanced actions may consult the [Base Object API](documentation/base_object_api.md) as well.

### Researchers

If you are a researcher interested in this line of research, it is recommended to consult first the [Research Guide](documentation/research_guide.md) which will provide commentary about the entire line of research, then the [User Guide](documentation/user_guide.md) which will discuss how the library is implemented, and [Optimizer Wrapper API](documentation/optimizer_wrapper_api.md) for underlying details of the algorithms.

## Getting Started

GNTS automatically tunes batch size during training by accumulating gradients based on their quality. The factory configures all schedules - just swap your optimizer initialization and train normally. This algorithm is continuously maintained as the example of the most productive implementation yet discovered. It has last been updated on 12/28/2025

Getting started with GQC is straightforward.

First, install the library from PyPi

```text
pip install torch-gqc
```

### Basic Usage

Suppose we have a classical learning loop, something like

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

The Gradient Norm Threshold Scheduling algorithm instead attempts to directly tune the ideal batch size using gradient accumulation. This involves several manipulations under the hood and is implemented as a function returning an optimizer wrapper and a schedule. The optimizer and schedules themselves implement all standard fields, and the optimizer passes through methods to the wrapped object. For this reason, this is composable with any existing optimizer and with the majority of frameworks.

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

Under the hood, this is implementing a ScheduleAnything schedule that sets the "gradient_norm_threshold" to follow a curve, from 0.95 to 0.25 by default. The system then accumulates gradients until the gradient norm is *below* this threshold. It should be kept in mind **lower numbers are more restrictive**. and the exact values can be edited. Please also keep in mind that due to lower numbers being more restrictive this uses an **inverse warmup** when controlling the norm threshold, though it uses a normal warmup for weight decay and loss.

### Custom Parameters

Sometimes the gradient norms are so large that the system crawls during warmup, despite being stable later, or sometimes you need a different schedule. These issues are correctable, but optional.

```python
from gradient_quality_control import OptimizerWrapperGNTS, make_gnts_with_cosine_annealing_schedule

...
train_loader = get_train_loader(batch_size = 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer, schedule = make_gnts_with_cosine_annealing_schedule(optimizer,
                                                             num_warmup_steps = 500,
                                                             num_training_steps = 10000,
                                                             norm_warmup_target=1.0, # 1.0 instead
                                                             norm_anneal_target=0.20, # A bit lower
                                                             initial_warmup_multiplier = 100.0 # Much higher than normal.
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

### Monitoring with Statistics

A wide variety of control statistics are also available by means of the statistics method, and a set of well-chosen vital statistics relevant to the optimizer and suitable for tqdm display or logging are available through .vital_statistics()

```python
from gradient_quality_control import OptimizerWrapperGNTS, make_gnts_with_cosine_annealing_schedule

...
train_loader = get_train_loader(batch_size = 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer, schedule = make_gnts_with_cosine_annealing_schedule(optimizer,
                                                             num_warmup_steps = 500,
                                                             num_training_steps = 10000,
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
    vital_statistics = optimizer.vital_statistics()
    pbar.set_postfix(vital_statistics)
```

### Distributed Training

For distributed training with DDP, FSDP, or other mechanisms you must tell the system what kind distribution is occurring. Are you replicating the model? Or sharding it?

```python
from gradient_quality_control import make_gnts_with_cosine_annealing_schedule

# For DDP (Data Parallel) - use "replicated"
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer, schedule = make_gnts_with_cosine_annealing_schedule(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000,
    distributed_mode="replicated"  # For DDP
)

# For FSDP (Fully Sharded Data Parallel) - use "sharded"
optimizer, schedule = make_gnts_with_cosine_annealing_schedule(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000,
    distributed_mode="sharded"  # For FSDP
)
```

In cases like MoEs where some parts are replicated and some parts are sharded, you will end up needing to use separate optimizers stepping the replicated and sharded parameters separately so the system knows the right reduction form to use. The rest of the training loop remains identical to the basic usage example.

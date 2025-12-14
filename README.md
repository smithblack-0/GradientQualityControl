# Gradient Quality Control

**Gradient Quality Control (GQC)** is a library that improves gradient quality by means other than datasource filtering before the gradients ever reach the optimizer. This library provides production-grade, drop-in optimizer wrappers implementing GQC algorithms via adaptive sampling. 

The solution is a new kind of **component** lying orthogonal to standard optimizers that preconditions the gradients to a higher quality before the optimizers ever observe them. These **Gradient Cleaner** wrappers dynamically vary batch size through gradient accumulation to maintain consistent gradient quality, significantly improving token sample efficiency during pretraining. **They operate in constant memory, are compatible with almost any pytorch optimizer, and require minimal training loop changes.**

## What do I do to use this?

This library requires proficiency in adding and lightly modifying torch training loops to use, and requires the ability to follow a guide. It is implemented as a minimally invasive optimizer-wrapper in pytorch. Jump down to getting started for full steps.

## How does it work?

The system is literally implemented as an optimizer-wrapper that takes over invoking zero_grad() and .step() from the user. The module then decides when to take a step, performing gradient accumulation and increasing gradient quality proportionally. Some Cleaners apply other tricks as well, but they all act as optimizer wrappers to improve gradient quality before the gradients hit the optimizers themselves. 

The main cleaner is currently the Gradient Norm Threshold Scheduler (GNTS) variety aims to keep the gradient norms below a particular threshold, and this threshold is then scheduled.

## Why would I want it?

Two main reasons

1) **It largely eliminates batch tuning as a hyperparameter**. The system works best when you choose a physical batch size that just reaches full gpu occupancy. GNTS and other controllers then maintain the same logical batch size largely invarient of the physical batch size, and the right hyperparameters are relatively robust across token numbers and model sizes.
2) **It gains you a little extra performance on well-tuned models**: Even when the model has already been tuned, the reactive nature of the GNTS controller ensures it draws more batches when the gradients gets noisy, which tends to improve training efficiency by a few percentage. 

Overall, the most promising outcome is perhaps not necessary the gains, but the ability to trust during research that the batch hyperparameter will be set to sane values and the model will automatically recover if the gradients get more noisy. It thus has a natural niche at small or medium scale labs or startups that cannot afford full tuning, but there is no reason not to use it on any single-device training run as far as we can tell. Large scale labs or companies will likely see less benefit, as Distributed Data training will tend to already push the batch size well above what GQC would have naturally chosen.

## Notable outcomes

Claimed results are preliminary and small-scale, and should be interpreted as suggestive rather than definitive. Results are currently listed in terms of the current generation of controller, GNTS.

| Event                                                                  | Outcome                             |
|------------------------------------------------------------------------|-------------------------------------|
| gpt2-small model setup to Karparthy tuned standards vs GNTS tuned | about a 5% gain in perplexity       |
| gpt2-small catastrophically mistuned batch size vs GNTS tuned     | about a 30% gain in perplexity      |

# For Practitioners

## Getting Started 

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

In GQS-AS, instead, we would directly control the step size and signal-to-noise ratio by demanding the gradient norm be a certain magnitude before stepping. Note when taking a mean of microbatch gradients extra batches tend to decrease the norms, which has warmup implications. We would also tend to reduce the physical batch size down to as low as achieves good gpu occupancy, as gradient accumulation will make it larger.

```python
from gradient_quality_control import OptimizerWrapperGNTS, get_norm_threshold_cosine_annealing_with_warmup

...
train_loader = get_train_loader(batch_size = 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_constant_schedule_with_warmup (optimizer, warmup_steps=500, ...)

# Optimizer wrapper intercepts schedule and automatically steps 
# when quality is high enough. Note we need to replace the built-in
# warmups as norms targets should actually start much higher than needed,
# not at zero as built-in solutions request.
optimizer = OptimizerWrapperGNTS(optimizer)
norm_scheduler = get_norm_threshold_cosine_annealing_with_warmup(optimizer,
                                                                num_warmup_steps = 500,
                                                                 num_training_steps = ...,
                                                                 start_norm = 0.8,
                                                                 end_norm = 0.2, # Where the schedule ends at
                                                                 )

for inputs, labels in train_loader:
    
    # Loss
    logits = model(inputs)
    loss = cross_entropy(logits, labels)
    loss.backward()
    
    # Optimization. IMPORTANT! No zero grad anymore, optimizer now takes care of that.
    optimizer.step()
    lr_scheduler.step()
    norm_scheduler.step()
```

Excellent logging and console usage is also supported; those using optimizer however should callbacks should consult the more detailed documentation in usage to know how to retrieve the callback returns. Instead, the step function in this library tells us whether the optimizer was stepped by the wrapper, and .statistics returns various statistics suitable for logging or console display.

```python
from gradient_quality_control import OptimizerWrapperGNTS, NormWarmupScheduler
from tqdm import tqdm

...
train_loader = get_train_loader(batch_size = 8)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_warmup_scheduler(optimizer, warmup_steps=500, ...)

# Optimizer wrapper intercepts schedule and automatically steps 
# when quality is high enough. In this configuration, we are using
# the start->end defaults of 1.0 -> 0.0. They are not perfect, but
# work well for small and medium models. 
optimizer = OptimizerWrapperGNTS(optimizer)
norm_scheduler = get_norm_threshold_cosine_annealing_with_warmup(optimizer,
                                                                 num_warmup_steps = 500,
                                                                 num_training_steps = ...,
                                                                 )

# Track optimizer step events
step_batches = []
num_batches_sampled = []

pbar = tqdm(train_loader, desc="Training")

for inputs, labels in pbar:
    # Stat draw comes before optimizer step so we do not  
    # clear num_draws out prematurely.
    stats = optimizer.statistics()

    # Loss
    logits = model(inputs)
    loss = cross_entropy(logits, labels)
    loss.backward()
    
    # Optimization
    stepped = optimizer.step()
    lr_scheduler.step()
    norm_scheduler.step()
    
    # Log when optimizer steps
    if stepped:
        step_batches.append(stats['batches'])  
        num_batches_sampled.append(stats['num_draws'])
    
    # Update progress bar
    pbar.set_postfix(stats)
```

Note that attaching the schedule to the OptimizerWrapperGNTS instead made it set the target gradient norm threshold; under the hood, we draw microbatches until noise cancels out sufficiently to meet that threshold. A cosine annealing from 1.0 to 0.2 is not atypical. This replaces the learning rate schedule by directly conditioning the gradients used to decide the step size instead. **The threshold is an upper bound on the gradient norm, not a lower bound**.

**Important: Norm scheduler warmup should be inverted from LR warmup**
- LR warmup: start low (0.0) → ramp up to peak
- Norm warmup: start high (example 5.0) → ramp down to target (1.0)
**Important: Generally, you should set your physical batch size to as small as is possible while reliably achieving gpu occupancy for best performance**

## Distributed Compatibility.

Largely, since these operate by gradient accumulation, distributed capacity should 'just work'. 1.0 will work with DDP and related, but minor adjustments to hyperparameter thresholds according to provided formulas are likely needed. For the moment, consider multiplying GNTS thresholds by the square root of the number of devices until stronger empirical rules can be explored. If anyone wants to sponsor a study into it, I am available. 

# For Researchers

Sufficient data has been gathered to state some conclusions about what this system is or is not doing. They are roughly as follows

1) **The primary effect GNTS has is to autotune batch size**: Going from a catastrophically mistuned case to a GNTS case illustrates this nicely. Getting rid of an optimization dimension, particularly in exploratory research, is very nice. The system requests the same logical batch size regardless of physical batch size as well.
2) **Some performance is gained from the reactive nature of the controller**: When starting from canonically solid and tuned defaults (Karpathy's GPT2-small replication) the performance gains are real but more minor, around 5-10%.
3) **The gradient noise scale did not corrolated properly with LLM performance**: We conjecture something in Adam is responsible, but the GNTS controller performed much better than the GNS controller. It looks like there are other reasons having isonorm gradients are beneficial to training.

The previous scaling law study will shortly be redone. At the moment, the effect is confirmed and real, but the envelope of usefulness and the empirical laws for scaling are unknown. If anyone is interested in working with me on research, contact me at chrisoquinn.2@gmail.com.

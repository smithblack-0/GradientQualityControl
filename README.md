# Introduction

**Gradient Quality Control (GQC)** is a training paradigm that improves gradient quality by means other than datasource filtering before the gradients ever reach the optimizer. Most of our algorithms do this by drawing additional samples adaptively, rather than relying on post-facto optimizer denoising mechanisms that primarily slow down training

This library provides research-grade, drop-in optimizer wrappers implementing GQC algorithms via adaptive sampling. These wrappers dynamically vary batch size through gradient accumulation to maintain consistent gradient quality, significantly improving token sample efficiency during pretraining. They operate in constant memory, are compatible with almost any pytorch optimizer, and require minimal training loop changes.

# Gradient Quality Control and Adaptive Sampling

Traditional mechanisms for GQC primarily focus on preprocessing data. However Adaptive Sampling GQC (AS-GQC) instead reacts to irregularities in training metrics themselves, and attempts to draw additional microbatches using gradient accumulation to increase the logical batch size  itself until a quality threshold is somehow passed. As these metrics are drawn directly from each microbatch, the system is reactive and can react to bad batches by drawing more samples to "anomaly smooth" the training stream.

We thus classify GQC-AS as orthogonal to optimizer theory. Optimizers take the gradients they have and do the best job possible with them; AS ensures we draw enough samples the noise level is constant to begin with.

# Notable outcomes

Notable outcomes so far showing some strengths and limitations include:

| Event                                  | Outcome                             |
|----------------------------------------|-------------------------------------|
| 50m model trained on 282m tokens       | 41% improvement in perplexity       |
| 50m test vs 800m control               | 5% improvement in perplexity at 50m |
| 50m model tried at various batch sizes | logical batch size largely the same |
| 50m test model on multiepoch task      | converged to a worse floor          |

No fine tuning has been tested yet. This tends to have much higher sample efficiency, but also may be sensitive to regularization.

# For Practitioners

## Getting Started 

Getting started with GQC is straightforward. We discuss a quickstart guide here.

First, install the library from PyPi

```text
[Todo]
```

Now, suppose we have a classical learning loop, something like

```python
optimizer = torch.optim.AdamW(model.parameter(), lr=lr)
scheduler = get_cosine_annealing_schedule(optimizer, ...)
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

In GQS-AS, instead, we would directly control the step size and signal-to-noise ratio by demanding the gradient norm be a certain magnitude before stepping. Note when taking a mean of microbatch gradients extra batches tend to decrease the norms, which has warmup implications.

```python
from gradient_quality_control import OptimizerWrapperGTNS, norm_scheduler_cosine_annealing

...

optimizer = torch.optim.AdamW(model.parameter(), lr=lr)
lr_scheduler = get_warmup_scheduler(optimizer, ...)

# Optimizer wrapper intercepts schedule and automatically steps 
# when quality is high enough. Attached schedules control the 
# gradient norm target!!!!
optimizer = OptimizerWrapperGTNS(optimizer)
norm_scheduler = norm_scheduler_cosine_annealing(optimizer, ...)

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
from gradient_quality_control import OptimizerWrapperGNTS, norm_scheduler_cosine_annealing
from tqdm import tqdm

...

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_warmup_scheduler(optimizer, ...)

# Optimizer wrapper automatically steps when quality is high enough.
# Attached schedules control the gradient norm threshold.
optimizer = OptimizerWrapperGNTS(optimizer)
norm_scheduler = norm_scheduler_cosine_annealing(optimizer, ...)

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

Note that attaching the schedule to the OptimizerWrapperGTNS instead made it set the target gradient norm threshold; under the hood, we draw microbatches until noise cancels out sufficiently to meet that threshold. A cosine annealing from 1.0 to 0.2 is not atypical. This replaces the learning rate schedule by directly conditioning the gradients used to decide the step size instead.

**Important: Norm scheduler warmup should be inverted from LR warmup**
- LR warmup: start low (0.0) → ramp up to peak
- Norm warmup: start high (example 5.0) → ramp down to target (1.0)

## Going deeper

Consult [Usage](documentation/usage.md) for information on using the various classes, the options available, and the intended usage paradigm.

# For Researchers

GQC-AS operates as a Sequential Binary Decision Controller: after each microbatch, the system decides whether gradient quality is sufficient to step, or whether to accumulate another batch.

**Key findings** (scoped to 50M-800M parameters, ~280M tokens):
- Models require ~1/3 the optimizer steps of standard training
- Models consistently beat their controls, and appear to auto-tune the logical batch size.
- Direct gradient magnitude control eliminates need for learning rate decay, and allows AdamW to train faster.
- Gradient Noise Scale does not accurately predict optimal step timing, and this is conjectured to be due to adam interactions. Adam instead appears to prefer isostep operation where the gradients consistently have the same magnitude.

**Detailed analysis, ablations, and theoretical discussion:** 

See [implementations](documentation/research/research_implementations.md) for a summary of what has been tested.
See [theory](documents/research/results_and_thoery.md) for a discusson of what the emperical results have uncovered, and what implications it may have for optimizer theory, model design, and more.

**Collaboration and Replication**

See the experiments folder at [experiments](examples) to view the research colabs used in the studies, replicate the results yourself, and draw your own conclusion. The "Budget" series of experiments can be reproduced in under 150$. Please credit this repository and the discussion inside, and switch to the formal paper when it comes out, when extending the results.

Anyone with compute resources, publication experience, or even work offers are suggested. See [collaboration](documents/research/collaboration.md) for details, and consider it the document kept up to date.

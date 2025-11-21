# Gradient Quality Control

**Gradient Quality Control (GQC)** is a training paradigm that improves gradient quality by means other than datasource filtering before the gradients ever reach the optimizer. Most of our algorithms do this by drawing additional samples adaptively, rather than relying on post-facto optimizer denoising mechanisms that primarily slow down training. This tends to significantly improve pretraining speed.

This library provides production-grade, drop-in optimizer wrappers implementing GQC algorithms via adaptive sampling. The solution is a new kind of **component** lying orthogonal to standard optimizers that preconditions the gradients to a higher quality before the optimizers ever observe them. These **Gradient Cleaner** wrappers dynamically vary batch size through gradient accumulation to maintain consistent gradient quality, significantly improving token sample efficiency during pretraining. **They operate in constant memory, are compatible with almost any pytorch optimizer, and require minimal training loop changes.**

Full understanding of the phenomenon is currently at a 'research-grade' level, but appears likely to scale nicely and, at minimum, is extremely beneficial when training small-scale models.

## What do I do do use this?

This library requires proficiency in adding and lightly modifying torch training loops to use.

You replace nothing. You still operate using your standard training loop, and add a component. Using the library is more akin to adding gradient clipping to a transformer without it than swapping a component such as replacing SGD with AdamW. Notably, as the underlying mechanism is just a special version of gradient accumulation, anything that can perform gradient accumulation is in theory compatible with these algorithms.

The system is literally implemented as an optimizer-wrapper that takes over invoking zero_grad() and .step() from the user. The module then decides when to take a step, performing gradient accumulation and increasing gradient quality proportionally. Some Cleaners apply other tricks as well, but they all act as optimizer wrappers to improve gradient quality before the gradients hit the optimizers themselves.

## I wandered off the internet when I heard about this. Can you provide some intuition?

First, understand this is just our best guess right now. We are still trying to understand the underlying mechanism. But this is what we think is happening.

We have been feeding our models with gasoline (gradients) that is 98% water and kludged together enough weird tricks our models to tolerate that. 

If we instead boil away the water in the first place, the engines (models) run faster. Even better we can probably make higher-precision engines that could not run on water-gasoline in the first place. Boiling away the water takes energy (compute), but we run so much faster it is worth it.

Sometimes the water becomes steam, actually adding power, but the effect varies depending on the rpm the engine is at  (stage of pretraining). So we need to change the ratio as training continues (gradient norm magnitude scheduling).

 Analogy is not guaranteed to work for all theory aspects. We will attempt to update this analogy as more information is uncovered.

##  Notable outcomes

Claimed results are preliminary and small-scale, and should be interpreted as suggestive rather than definitive. However, the success of the effect across multiple controller variations should be interpreted as 'highly suggestive' and extends many scaling priors. Work to provide full experimental harnesses for all axes at a **production** level is ongoing. Feel free to get involved!

Notable outcomes so far showing some strengths and limitations include:

| Event                                  | Outcome                             |
|----------------------------------------|-------------------------------------|
| 50m model trained on 282m tokens       | 41% improvement in perplexity       |
| 50m test vs 800m control               | 5% improvement in perplexity at 50m |
| 50m model tried at various batch sizes | logical batch size largely the same |
| 50m test model on multiepoch task      | converged to a worse floor          |


Stated for lay audiences: The same model ends up much better, and the results are significant enough to beat a model 16x larger.

There are limitations.

* No fine tuning has been tested yet.
* We may be improving small and midscale model behavior to match large model behavior; it is unknown whether this scales.
* Some of these results were found using the old, less effective controller, but as part of the same research push.

# For Practitioners

## Getting Started 

Getting started with GQC is straightforward. We discuss a quickstart guide here.

First, install the library from PyPi

```text
pip install torch-gqc
```

Now, suppose we have a classical learning loop, something like

```python
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

In GQS-AS, instead, we would directly control the step size and signal-to-noise ratio by demanding the gradient norm be a certain magnitude before stepping. Note when taking a mean of microbatch gradients extra batches tend to decrease the norms, which has warmup implications.

```python
from gradient_quality_control import OptimizerWrapperGNTS, get_norm_threshold_cosine_annealing_with_warmup

...

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

## Distributed Compatibility.

Largely, since these operate by gradient accumulation, distributed capacity should 'just work'. 1.0 will work with DDP and related, but minor adjustments to hyperparameter thresholds according to provided formulas are likely needed. Advanced users should consult [Usage](documentation/usage.md) for the necessary formulas.

## Limitations

Fine-tuning performance is unknown, but likely to be suboptimal without significant retuning of regularization. Scaling behavior appears promising but has not been tested above 800m parameters. We conjecture that the norm schedule should be reduced in as you would clipping rules, but cannot prove it right now.

# For Researchers
**Those interested only in how to use this library should stop reading here**

GQC-AS operates as a Sequential Binary Decision Controller: after each microbatch, the system decides whether gradient quality is sufficient to step, or whether to accumulate another batch.

## Why does this work?

We don't know. It just does. Those who want a more formal verison of that are encouraged to jump into [theory](documents/research/results_and_thoery.md), those who want a simple summary can just keep reading. 

Generation 1 analysis revealed paradoxes:

- Control models show < 1° angle between Adam momentum and raw gradients on control cases (near-perfect alignment)
- Yet accumulation can reduce gradient norms by a factor of 20, suggesting massive amounts of noise.
- Generation 1 fitting produced data exponent β ≈ 0.35 (Kaplan tradition) WITHOUT hyperparameter tuning - normally this requires extensive search
- The fit was unstable but suggestive of improved scaling behavior
- Naive gaussian error theory with Adam Moments analysis suggests reducing noise but taking more steps should balance out; it clearly did not.

The mathematics say with Adam more steps at higher noise is equivalent to less steps at higher lower noise. The empirics say removing the noise helps tremendously despite the signal already being present. We are much more confident than not that noise is being reduced than not and that is is helping and measurable, but paradoxically it is detectible by one means but not by another.

One notable possible explanation is the reactive nature of most of the tests: Difficult batches usually cause more draws. This is the case with the GNS, GNTS, and MHT mode. We call this phenomenon **Anomaly Smoothing**. Given what has been observed, there is also a large likelyhood having gradients that are consistently the same magnitude is extremely beneficial as well. But if anomaly smoothing was the only effect, why did ensuring consistent gradient norm magnitudes in GNTS help too?

Key unknowns:
- What does accumulation actually do to gradient-momentum alignment? Where is the excess magnitude we are cancelling away living?
- Is the anomaly smoothing the primary driver of the observed effects? The constant gradient magnitudes? The extra batches? Something else?
- What explains the incongruency between angle measures and magnitude measures?- Is this an Adam-specific phenomenon or general to adaptive optimizers?
- If we are removing noise, do approximate second order optimizers, such as Shampoo and K-FAC, do better with better curvature estimates?

This is active research with incomplete theory. The results are too strong to ignore, but we cannot yet explain why they occur.

## More details


See [implementations](documentation/research/research_implementations.md) for a summary of what has been tested.
See [theory](documents/research/results_and_thoery.md) for a discusson of what the emperical results have uncovered, and what implications it may have for optimizer theory, model design, and more.

See the experiments folder at [experiments](examples) to view the research colabs used in the studies, replicate the results yourself, and draw your own conclusion. The "Budget" series of experiments can be reproduced in under 150$. Please credit this repository and the discussion inside, and switch to the formal paper when it comes out, when extending the results.

Anyone with compute resources, publication experience, or even work offers are suggested. See [collaboration](documents/research/collaboration.md) for details, and consider it the document kept up to date.

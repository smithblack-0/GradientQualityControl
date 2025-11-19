# Gradient Quality Control and Adaptive Sampling
**Gradient Quality Control (GQC)** is a training paradigm that improves gradient quality by means other than datasource filtering before the gradients ever reach the optimizer. Most of our algorithms do this by drawing additional samples adaptively, rather than relying on post-facto optimizer denoising mechanisms that primarily slow down training. This tends to significantly improve pretrainiing speed.

This library provides research-grade, drop-in optimizer wrappers implementing GQC algorithms via adaptive sampling. These wrappers dynamically vary batch size through gradient accumulation to maintain consistent gradient quality, significantly improving token sample efficiency during pretraining. **They operate in constant memory, are compatible with almost any pytorch optimizer, and require minimal training loop changes.**

# Notable outcomes

Notable outcomes so far showing some strengths and limitations include:

| Event                                  | Outcome                             |
|----------------------------------------|-------------------------------------|
| 50m model trained on 282m tokens       | 41% improvement in perplexity       |
| 50m test vs 800m control               | 5% improvement in perplexity at 50m |
| 50m model tried at various batch sizes | logical batch size largely the same |
| 50m test model on multiepoch task      | converged to a worse floor          |

No fine tuning has been tested yet. This tends to have much higher sample efficiency, but also may be sensitive to regularization. A notable limitation is that we may be improving small and midscale model behavior to match large model behavior, rather than improving scaling laws as a whole.

## Explain it like I am 5.

We have been feeding our models with gasoline (gradients) that is 98% water and kludged together enough weird tricks our models to tolerate that. 

If we instead boil away the water in the first place, the engines (models) run faster. Even better we can probably make higher-precision engines that could not run on water-gasoline in the first place. Boiling away the water takes energy (compute), but we run so much faster it is worth it.

Sometimes the water becomes steam, actually adding power, but the effect varies depending on the rpm the engine is at  (stage of pretraining). So we need to change the ratio as training continues (norm magnitude scheduling)

Theorists and angry people on the internet are encouraged to jump down into the "for researchers" section. The above noise to signal numbers were empirically measured using a reproducible procedure, though at small scale only, so actual numbers may vary for your application. Analogy is not guaranteed to work for all theory aspects.

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
from gradient_quality_control import OptimizerWrapperGNTS, NormWarmupScheduler

...

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_constant_schedule_with_warmup (optimizer, warmup_steps=500, ...)

# Optimizer wrapper intercepts schedule and automatically steps 
# when quality is high enough. Note we need to replace the built-in
# warmups as norms targets should actually start much higher than needed,
# not at zero as built-in solutions request.
optimizer = OptimizerWrapperGNTS(optimizer)
norm_scheduler = get_cosine_annealing_schedule(optimizer, warmup_steps=500, ...)
norm_scheduler = NormWarmupScheduler(norm_scheduler, warmup_steps= 500)

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
# when quality is high enough. Note we need to replace the built-in
# warmups as norms targets should actually start much higher than needed,
# not at zero as built-in solutions request.
optimizer = OptimizerWrapperGNTS(optimizer)
norm_scheduler = get_cosine_annealing_schedule(optimizer, warmup_steps=500, ...)
norm_scheduler = NormWarmupScheduler(norm_scheduler, warmup_steps= 500)    

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

Note that attaching the schedule to the OptimizerWrapperGNTS instead made it set the target gradient norm threshold; under the hood, we draw microbatches until noise cancels out sufficiently to meet that threshold. A cosine annealing from 1.0 to 0.2 is not atypical. This replaces the learning rate schedule by directly conditioning the gradients used to decide the step size instead.

**Important: Norm scheduler warmup should be inverted from LR warmup**
- LR warmup: start low (0.0) → ramp up to peak
- Norm warmup: start high (example 5.0) → ramp down to target (1.0)

## Going deeper

Consult [Usage](documentation/usage.md) for information on using the various classes, the options available, and the intended usage paradigm. 

The underlying principle of operation is you gradient accumulate over multiple steps, until the gradient norms are below a threshold. This is guaranteed to happen as a mean of noisy vectors shrinks in magnitude.

The threshold is then scheduled to ensure that in early traing we accept a lot of noise, but by late training we accept little.


## Limitations

Fine-tuning performance is unknown, but likely to be suboptimal without significant retuning of regularization. This system does not function correctly over multiple epochs when batches are not, in fact, independent and may be converging to a worse floor. Scaling behavior appears promising but has not been tested above 800m parameters. We conjecture that the norm schedule should be reduced in as you would clipping rules, but cannot prove it right now.

# For Researchers

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

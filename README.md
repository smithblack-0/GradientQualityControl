# Gradient Quality Control

**Gradient Quality Control (GQC)** is a library that improves gradient quality by means other than datasource filtering before the gradients ever reach the optimizer. This library provides production-grade, drop-in optimizer wrappers implementing GQC algorithms via adaptive sampling. 

The solution is a new kind of **component** lying orthogonal to standard optimizers that preconditions the gradients to a higher quality before the optimizers ever observe them. These **Gradient Cleaner** wrappers dynamically vary batch size through gradient accumulation to maintain consistent gradient quality, significantly improving token sample efficiency during pretraining. **They operate in constant memory, are compatible with almost any pytorch optimizer, and require minimal training loop changes.**

## What do I do to use this?

This library requires proficiency in adding and lightly modifying torch training loops to use, and requires the ability to follow a guide. It is implemented as a minimally invasive optimizer-wrapper in pytorch. Jump down to getting started for full steps.

## How does it work?

The system is literally implemented as an optimizer-wrapper that takes over invoking zero_grad() and .step() from the user. The module then decides when to take a step, performing gradient accumulation and increasing gradient quality proportionally. Some Cleaners apply other tricks as well, but they all act as optimizer wrappers to improve gradient quality before the gradients hit the optimizers themselves. 

The main cleaner is currently the Gradient Norm Threshold Scheduler (GNTS) variety aims to keep the gradient norms below a particular threshold, and this threshold is then scheduled.

## Why would I want it?

Four main reasons, all of them awesome!

1) **It largely eliminates batch tuning as a hyperparameter**. The system works best when you choose a physical batch size that just reaches full gpu occupancy. GNTS and other controllers then maintain the same logical batch size largely invariant of the physical batch size, and the right hyperparameters are relatively robust across number of tokens, physical batch size, and model size. Since these are extraordinary claims, feel free to consult research for details.
2) **It gains you a little extra performance on well-tuned models**: Even when the model has already been tuned, the reactive nature of the GNTS controller ensures it draws more batches when the gradients gets noisy, which tends to improve training efficiency by a few percentage. 
3) **It works out of the box in DDP distributed environments**: As long as you ensure gradients have been averaged together before the optimizer wrapper is called to step, the system does not care whether it is in a distributed environment or not. It will still automatically notice the reduction in noise from the extra devices, and steps earlier. 
4) **It naturally adapts to noise in exploratory research**: Since the system is reactive to noise in the gradients, it should allow much faster iteration during exploratory research in novel models which may implement layers that increase the noise level shifting where the ideal batch size is. This makes it significantly easier to avoid false negatives when you do not have funding for hyperparameter tuning.

Overall, the most exciting consequence may be the democratization effect. It is possible for small groups to do research or training that used to require a much larger budget, while not significantly increasing the capacities at the frontier end of the scale, increasing research churn and efficiency. This was, in fact, the primary design goal of this line of research.

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

## What level of support does this have?

It is intended to be usable at an enterprise level. Feel free to submit bugs on the github page. No API changes will occur without at least 1 year of deprecation warnings. The GNTS controller is done, as is the SBC controller.

Some minor amount of development may still occur elswhere with more controller variants, however they will be found in the experimental folder. 

Support of this library will occur for no less than two years (2027). If someone at torch is interested, I would be thrilled to move a version of it onto the pytorch main branch.

## Is this distributed compatible?

Partially. DDP and other distributed data systems that replicate the model across devices and feed data through faster should be fully compatible. However, any system that places different groups of parameters on different models largely loses compatibility.

If anyone is interested in supporting an additional use case, put in a features request. Please note that depending on how complex it is I may need additional support to get the job done. 


# For Researchers

If you are not interested in how this works, feel free to stop reading here.

## Introduction

This entire mechanism is something I have been chasing down for awhile now, after noticing you might be able to autotune batch size by detecting variance at the loss. Now, that we are at the GNTS formulation, we are trying to optimize the following objectives.

Given a tuning of hyperparameters at one configuration and size, involving using a Gradient Cleaner and setting the start threshold, we would like it to be the case:

* Changing the logical batch size has only a minor and recoverable effect on training
* Changing the model size has only a minor and recoverable effect on training
* Changing the number of tokens trained with has only a minor and logical effect on training.
* Changing the dataset has only a small and minor effect on training.

These reasons necessitated the reactive controllers that the Gradient Quality Control library tends to focus on, so if a change causes more noise it compensates. Basically, the point is "Tune once, deploy everywhere". The question is how close we have gotten.

## Notable outcomes

All listed results are in terms of the current generation of controller the Gradient Threshold Norm Scheduler. The control setup was based on Karparthy's GPT2-Small replication, with a learning rate of 6e-4, a batch size of 64, openwebtext, and a sequence length of 1024. 

It should be noted that due to budget limitations none of these runs were to chinchilla optimality, and rather usually used 150M tokens unless listed otherwise. As such, read perplexity gains as 'this trains faster' rather than 'this will converge more deeply', as the latter cannot (yet) be proven.

| Event                                                                                 | Outcome                                                      |
|---------------------------------------------------------------------------------------|--------------------------------------------------------------|
| 1) Gpt2-small Hyperparameter Empirical probe                                          | Best performer is from 0.95->0.25  norm schedule             |
| 2) gpt2-small model setup to Karparthy tuned standards vs GNTS tuned                  | About a 16% gain in perplexity (53.2->44.4) in same time     |
| 3) gpt2-small catastrophically mistuned batch size vs GNTS tuned                      | about a PLACEHOLDER% gain in perplexity                      |
| 4) gpt2-small at physical batch size of 4, 8, 16, 32                                  | Usually requested the same logical batch size, and within 5% |
| 5) gpt2-small norm threshold hyperparameters seek varying token counts                | Same hyperparameters at both sizes                           |
| 6) gpt2-medium norm threshold hyperparameter seek                                     | Same hyperparameters as gpt2-small                           |
| 7) gpt2 small and medium scaling law fit (parameters, tokens) with GNTS as Chinchilla | Pending...                                                   |
|
| ---------------------------------------------------------------------                 | --------------------------------------------------           |

## Reproducibility
 
For those interested, the notebooks are provided for reproduction purposes. If run in their original configuration, with a gpu attached in colab, they should load the results and plot them. To reproduce them yourself, setup a huggingface repository, hook it up in the logging and filesystem section, add your secret as a token in colab, and give it awhile. If you use other devices, it will naturally go faster or slower. It should require only a few minor changes to get it working on jupyter systems as well, namely you will have to handle your own secrets yourself and remove the google.colab import. The excellent, self-developed SOARS open-source testing system is used, so the notebooks should be quite navigable; I would be interested on feedback on if I should release it as a standalone tool. If you use colab, note you may have to restart it a number of times due to timeouts, but the checkpointing system has your back.

| Event | Time for reproduction | Cost for Reproduction          | Device      | Colab Link                                                                                                                                                          |
|-------|-----------------------|--------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | 31 Hours              | About $5.2, 52 compute credits | L4          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SVjIFrASLuH3GaglVi4soTGbM6Seid9N?usp=sharing) | 
| 2     | 31 Hours              | About $5.2, 52 compute credits | L4          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SVjIFrASLuH3GaglVi4soTGbM6Seid9N?usp=sharing) | 
| 3     | Placeholder           | Placeholder                    | Placeholder | Placeholder                                                                                                                                                         |
| 4     | 3.5 Hours             | About $2.5, 25 compute credits | A100        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A7tMsPlXf5z5XgdhGHNGc4SVnWnUQbOQ?usp=sharing) |
| 5     | 31 Hours              | About $5.2, 52 compute credits | L4          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SVjIFrASLuH3GaglVi4soTGbM6Seid9N?usp=sharing) | 
| 6     | 4.5 Hours             | About $4.1, 41 compute credits | A100        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gZDOsV06rUtzBgwVD4ILlV53MYJ2uQ7D?usp=sharing) | 

## Theory

### Taxonomy

The kind of mechanism that GNTS is is largely missing a formal class in the literature. We define it to be a **Gradient Preprocessor**. These are distinguished from optimizers by the ability to draw more samples before stepping, and otherwise control when the gradients are a high enough quality to step. This places them **orthogonal** to the standard optimizer, as you can mix and match preprocessors and optimizers at will.

Furthermore, we identify that the decision of whether to draw another sample or continue accumulation is a control decision. As such, we define the taxonomy of the algorithm to be a **Sequential Binary Decision Algorithm** deciding on each draw whether to step or draw more samples. Under this taxonomy, the **Gradient Cleaner** taxonomy, which is the primary taxonomy this library lives in, is the union of the Sequential Binary Decision Controller and Gradient Preprocessor categories: It is responsible for deciding when to step in order to maintain a level of gradient quality

### GNTS operational theory

The operational theory of the GNTS subsystem and the noise cancellation mechanism behind is largely covered [here](documentation/research/LLM%20Gradient%20Noise%20Theory.md). Note that the initial presumption that the gradient noise scale was broken may have been an artifact of an incorrect cleaner implementation, and can be probed again in future research. 

However, as a brief summary, we show that

1) averaging gradients should tend to reduce their magnitudes
2) This makes the magnitude of the gradients a proxy for the signal quality of the gradient
3) This means if we wait to step until the gradient magnitude goes below a norm, we have a reactive SBDC system that recovers from anomalies automatically and operates in an isonorm gradient magnitude regime
4) By then scheduling that threshold, we can allow noisy updates during early training, and clean ones later on, entirely replacing the cosine annealing schedule on the learning rate as the length of the gradient shrinks instead.

It is also  suspected the standard adam optimizer has more favorable second moment characteristics when operating in this isonorm magnitude regime, providing a bit of an additional boost, but this is speculation for the moment.

## Conclusions

We will divide this up into three epistemological categories

* Concrete: It does this. There is enough evidence to be certain
* Provisional: Subject to a few constraints, it does this
* Speculative: When we bring in the larger context, we have reason to believe it does this

### Can a reactive Gradient Cleaner improve the performance of a model over a large variety of regimes?

Yes. This is Concrete. We have seen this effect across models sizes, across logical batch sizes, across varying physical batches, and several generations of controllers. It helps early and mid training a lot as a minimal statement. It is almost certain this also extends to the large scale regime, with a catch

### Does this make training 'tune once and walk away'

And this is that catch. We know within an order of magnitude or so it is tune once and walk away, making this Concrete. But whether that tuning holds properly over several order of magnitudes worth of change is unknown. We believe it will, but must nonetheless mark this Speculative for lack of data.

### Does physical batch size actually matter once using GNTS?

Mostly no. The best choice is likely to choose the smallest batch that gives you full gpu occupancy. Too small a batch does, eventually, however produce a slowdown for unknown reasons. Nonetheless, for any 'reasonable' batch size the system will choose an effective logical batch size, within the tested domains. 

### Is this compatible with Distributed Training

Partially, Concrete:

Since distributed gradient technologies average the gradients before taking an optimizer step, so long as they support gradient accumulation they should always work just fine with the system. Systems like Mixture of Experts that distributed the parameters across various devices are not currently supported, however. The necessary addition would be to find a way to take the gradient norm of the entire model, and feed that into the algorithm. For those interested, I know a few extensions that might work.

## Open Questions

### Scientific 

* What does the late-phase convergence behavior look like? Does it depend on physical batch size? Does it converge any better using GNTS, or worse?
* What is going on in Adam's step size? Do other optimizer variants, like Shampoo or K-FAC, that need really clean gradients do better using this system?
* Can we develop a more thorough probe of ideal hyperparameter behavior with increasing number of tokens? With scale? 
* How does this behave as you scale up to much larger sizes? What are the scaling laws? Does it really 'just work'?
* What effect if any does changing the dataset have? Trying a wide variety of models?

### Operational

* Can I get someone to fund a scaling study to larger sizes?
* Can I connect to mentors to help maintain this long term?
* Can I get long term research report? This would go a lot faster with a training budget greater than $50 a month.
* Is there enough interest in an distributed model formulation I should code in a callback to support it? 
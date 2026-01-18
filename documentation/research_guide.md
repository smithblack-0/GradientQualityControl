# Research Guide

This guide provides an overview on the lines of research which have been pursued so far, what has been found, what theories have been developed, and what results have popped out. It is a summary of the research effort so far.

## Overview

This research began as a 'let's try it' effort to automate batch sizing that produced an artifact that suggests automatic batch size tuning may not be impossible and may produce significant gains. More importantly, it was speculated it could significantly accelerate small lab research by automating this form of hyperparameter probing.

The research is roughly divided into three "generations"

1) **Generation 1**: The initial line of inquiry, which surfaces the Metric Hypothesis Test formulation. Seeing a perplexity jump of 40% was enough to get attention; figuring out that the system was fairly invariant over physical batch size changes illustrated additional potential. Code quality was a bit lower.
2) **Generation 2**: Development of the SOAR research harness, and initial implementation of this library. More rugged exploration of possible confounding factors. Identification of 40% results as being indicative of misconfiguration; identification of more like 20% results as being typical;
3) **Generation 3**: Current generation (1/1/2025). It was realized weight decay was drastically misconfigured during generation 2, and more complex scheduling was needed. ScheduleAnything was deployed, and the Gradient Quality Control library is rebuilt.

This guide provides commentary on the research line and design decisions behind the Gradient Quality Control library.

## Navigation

- **[README](../New%20Readme.md)** - Installation and quick start
- **[User Guide](user_guide.md)** - Usage patterns and library overview
- **[Optimizer Wrapper API](optimizer_wrapper_api.md)** - Detailed wrapper specifications
- **[Wrapper Factories API Guide](api_guide.md)** - Pre-configured factory functions
- **[Base Object API](base_object_api.md)** - Abstract base class for extending

---

## Generations Guide

### Generation 1

Generation 1 consisted primarily of ad-hoc experiments to prove the effect was real, and focused around the Metric Hypothesis Test formulation. When faced with a 40% perplexity gain, and a strong enough effect that a 100m parameter model can outperform a 800m parameter one under isotoken budgets, one expects there was a confounding factor. It concluded, in brief

1) This effect is real. Reacting to noise can autotune batch size, it operates across datasets and models, and it gains performance.
2) The control was dramatically mistuned, and needs to be redone using a tuned default. Karpathy's GPT2 small replication was chosen for the next generation.
3) There are several control loops. Of them, GNTS is dramatically better and the next generation should use it.

Overall, the situation was the effect was real, there was a confounding factor, but it could not explain all of the performance. Notably, a custom model was used for most of the exploration. Generation 2 would thus be more rigorous. A brief summary of interesting results include:

| Event                                  | Outcome                             |
|----------------------------------------|-------------------------------------|
| 50m model trained on 282m tokens       | 41% improvement in perplexity       |
| 50m test vs 800m control               | 5% improvement in perplexity at 50m |
| 50m model tried at various batch sizes | logical batch size largely the same |
| 50m test model on multiepoch task      | converged to a worse floor          |

### Generation 2

Generation 2 was characterized by the development of the Gradient Quality Control research library and the development of the SOARS test harness; in other words, largely, it can be characterized by an increase in rigor and comparison to standard models. It also started from Karpathy's GPT2 replication for control behavior.

It should be noted that due to budget limitations none of these runs were to chinchilla optimality, and rather usually used 150M tokens unless listed otherwise. As such, read perplexity gains as 'this trains faster' rather than 'this will converge more deeply', as the latter cannot (yet) be proven.

| Event                                                                  | Outcome                                                      |
|------------------------------------------------------------------------|--------------------------------------------------------------|
| Gpt2-small Hyperparameter Empirical probe                              | Best performer is from 0.95->0.25  norm schedule             |
| gpt2-small model setup to Karpathy tuned standards vs GNTS tuned       | About a 16% gain in perplexity (53.2->44.4) in same time     |
| gpt2-small at physical batch size of 4, 8, 16, 32                      | Usually requested the same logical batch size, and within 5% |
| gpt2-small norm threshold hyperparameters seek varying token counts    | Same hyperparameters at both sizes                           |
| gpt2-medium norm threshold hyperparameter seek                         | Same hyperparameters as gpt2-small                           |

Generation two can said to have concluded when it was realized in transitioning from generation one to generation two a mistake was made handling weight decay. The entire generation two series was done with weight decay that did not actually reduce as training proceeded, as the learning weight was warmed up to a constant then stuck there, and the length of the gradients were controlled directly instead. This was only identified as an issue when replications across larger models were attempted; a larger model should never perform worse than the smaller one unless something is drastically misconfigured, as indeed it was. 

### Generation 3

The third generation is the one currently under development. It is built on top of the ScheduleAnything library, which has been implemented and spun out of the Gradient Quality Control library to allow examination of generalized scheduling. The abstraction has seen significant enough usage it is deemed worth it.

No results yet exist, but a comprehensive library rebuild is underway. As part of this, distributed operation is being incorporated as a primary concern, we are using documentation driven development, and will be completing the rebuild shortly.

## Research Notes

Various bits of research theory have been placed here.

### What is a solution?

We desire to optimize the ideal batch size using gradient accumulation. But what does it mean to have an optimal solution? Naively, it means to have good performance, but under what invariants?

We establish as the desired contracted invariants:

1) Maintains largely the same logical batch size regardless of the underlying physical batch size. 
2) Seeks out and maintains, as best as possible, the best logical batch size tuning when the same model is scaled up or down in size.
3) Seeks out and maintains, as best as possible, the best logical batch size tuning regardless of the underlying dataset.
4) Support this with defaults that largely 'just work' for common LLM pretraining activities.
5) React to low quality data and increase the batch size to compensate, and is fast enough to work viably

The algorithm that best balances these tradeoffs is judged to be the best one.

### Initial Theory of MHT operation

The initial metric hypothesis test formulation used loss as a metric, drew losses during gradient accumulation, and stepped only when a formal hypothesis test has cleanly pinned down what the loss actually was. The hope was we could then use that, with a one-time tuning, then generalize to giving reasonable performance across a variety of conditions. This worked, however a better formulation has since been developed. At this phase, nothing was scheduled. Both a loss and a gradient norm version were tested, although the norms had yet to be scheduled and generally performed worse. A reactive system was already a primary design requirement, and the whole thing started as 'let's solve this hyperparameter issue'.

### Gradient Noise Scale

After the viability of the concept of an adaptive controller was observed, the literature was consulted in an attempt to produce a more optimal formulation. This failed. With Adam, and using the GNS scale, the system proved unviable. Training proceeded between regions where the threshold was too loose to too strong, and it was clear these thresholds changed as training proceeded.

### Development of Gradient Magnitude Ratio

This meant the theory was broken. I am not adverse to making new theory, and that is exactly what I did. This lead directly to the GNTS controller generation defining generation 2 and 3. 


### Core Theory: Gradient Noise as Vector Cancellation

What do you do when your theory breaks? I am trained for this from my physics degree. I go back to the basics. I do so with the presumption:

* I want to directly measure the generalization signal in training gradients.
* I want to be able to do so in a manner that is fast and efficient.
* I want to be able to train LLMs to produce the best generalization behavior possible; Information Retrieval is strictly out of scope.

### Scope Of Theory: What is Generalization?

If I am to measure something as nebulous as how much 'generalization' signal exists, I must first have some idea of what it is in the first place. What is generalization, and how does it behave in LLM training? As a first approximation, we might say:

* **Generalization is the ability to apply a common strategy in a wide variety of situations**

Under this assumption, a generalization strategy is useful more or less everywhere. *Aha*. Usefully, this has a physical meaning in terms of gradient vectors. If we agree training signals useful to generalization are strategies that work everywhere, then in theory the mean of the gradient vectors across all training examples should produce a 'perfectly general' training signal within the training distribution.

 We conclude as priors for the formal mathematics that:

* The best generalization signal in the gradients is the mean of the gradients across all examples
* The task of any controller is to find the best balance between improving gradient quality and taking more steps.
* A way of even approximating how close we are to the true signal would be invaluable.
* GNS was an attempt to find this balance. It however failed for small-scale LLMs.

It was under these conditions we began to reexamine noise theory and gradient norms.

### Gradient Noise Model

To build a measurement, we first need a mathematical framework for how noise behaves when we sample gradients.

At training step $t$, we model each minibatch gradient as a random draw from a multivariate Gaussian distribution:

$$G_{tn} = \nabla L_t(P_n) \sim \mathcal{N}(\mu_t, \Sigma_t)$$

Here $\mu_t$ represents the true mean gradient over the training distribution, $\Sigma_t$ captures the batch-to-batch covariance, and $n$ indexes individual batch draws.

When we average $N$ independent gradient samples, standard Gaussian statistics gives:

$$\bar{G}_t = \frac{1}{N} \sum_{n=1}^{N} G_{tn} \sim \mathcal{N}\left(\mu_t, \frac{\Sigma_t}{N}\right)$$

The covariance scales as $1/N$. Informally speaking, the noise has mean zero and so vanishes over infinite samples - only signals that consistently point in the same direction express themselves in the mean.

### The Measurement Problem

In theory, the ideal measurement would be straightforward: compute the true mean gradient $\mu_t$, then measure the similarity (perhaps via cosine similarity or Euclidean distance) between our current gradient estimate and this true value. This approach has an obvious problem: we cannot compute $\mu_t$ during training. Computing the true mean would require evaluating gradients across the entire training distribution, which is exactly what we're trying to avoid by using minibatches in the first place. However, there is an approach which is viable in high-noise regimes. When the covariance matrix tends to be much larger than the mean, taking a mean of errors tends to make the length of the individual vectors shrink! This means the mean gradient norm is usable as a detection mechanism, and the ratio of the original to final gradient norm roughly bounds the error.

### The Tractability Problem

We've identified an observable signal - averaging gradients reduces their expected magnitude. But we still need to formalize this into something computable that quantifies signal quality.

We cannot directly compute a signal-to-noise ratio (SNR) because we don't have access to the true signal $\mu_t$ or the noise covariance $\Sigma_t$. However, we can bound the SNR using quantities that are measurable during training: the expected single-batch gradient norm and the norm of averaged gradients.

Define the **Gradient Magnitude Ratio (GMR)** at step $t$ as:

$$\text{GMR}_t = \frac{\mathbb{E}_n[\|G_{tn}\|]}{\|\bar{G}_t\|}$$

where $\mathbb{E}_n[\|G_{tn}\|]$ is the expected magnitude of single-batch gradients and $\|\bar{G}_t\|$ is the magnitude after averaging $N$ samples.

This ratio directly quantifies the magnitude reduction due to noise cancellation. If $\text{GMR}_t = k$, then single-batch gradients have $k$ times the magnitude of the averaged gradient, meaning approximately $(k-1)/k$ of the single-batch magnitude was noise that cancelled away.

In practice, this can be estimated during training by drawing multiple batches, computing the mean of their individual norms, and dividing by the norm of their average. For example, drawing 100 batches and computing:

$$\text{GMR}_t \approx \frac{\frac{1}{100}\sum_{n=1}^{100} \|G_{tn}\|}{\left\|\frac{1}{100}\sum_{n=1}^{100} G_{tn}\right\|}$$

provides a reasonable estimate of current gradient noise levels and overall training health.

### The Controller Problem

While GMR provides the ideal measurement of signal-to-noise ratio, it does not provide an ideal controller. At certain stages of training, noise is tolerable or even beneficial; taking more optimizer steps at higher noise can outperform fewer steps at lower noise. Indeed, a controller tested which attempted to always get within a certain threshold of $\mu$ performed very poorly. Nonetheless, lower noise is required late in training, a task traditionally achieved using learning rate schedules.

To accommodate both necessary conditions, we can use a controller based on gradient norms.  Given two mean gradients produced by different amounts of batch accumulation with the same training parameters, the gradient with the lower norm (more accumulation) has higher signal-to-noise ratio. This follows from the noise cancellation mechanism: more averaging removes more noise, reducing the norm closer to ||μ_t||.

The control theory becomes straightforward: set a gradient norm threshold and accumulate batches until the mean gradient norm drops below it. This is exactly what the GNTS controller does.  By scheduling this threshold over training, we directly control the effort-quality tradeoff - higher thresholds early (more steps, higher noise) and lower thresholds late (cleaner gradients). This even entirely replaces learning rate scheduling, as it directly controls the relevant gradient feature - step size - that we were controlling implicitly through learning rate schedules.

The gradient norm directly encodes noise level through the cancellation mechanism - it is not an arbitrary heuristic but a measurement associated with the quantity we care about. It is packaged in a way that lets us demand more quality according to a schedule. It also provides a reactive mechanism to noise, as norm spikes are responded to by drawing extra batches instead, and keeps per-parameter gradient variance consistent which is optimal for optimizers such as Adam.

# Open Questions

* What does the late-phase convergence behavior look like? Has the issue been corrected now that weight decay is removed?
* What is going on in Adam's step size? Do other optimizer variants, like Shampoo or K-FAC, that need really clean gradients do better using this system?
* Can we develop a more thorough probe of ideal hyperparameter behavior with increasing number of tokens? With scale? 

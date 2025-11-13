# Introduction

We briefly set the stage with classic optimizer theory, and a brief description of what was tried. 

## Scope of rigor

It should be noted conclusions in this theory section are preliminary. The Scope Of Rigor is within the 50m-800m model regime, limited to pretraining on only 280m tokens or so. C4 was the primary training dataset though pile and wikitext were tested. Additionally, LLama, Mistral, and GPT2 models were examimed.

## Classic Optimizer Theory

Minibatch Optimizer theory is classically grounded in imagenet experiments using stochastic gradient descent, and generally identified two ways to reduce and manage gradient noise:

* We can use a bunch of noisy gradient samples, and the errors average out over time in accordance to classical gaussian error theory
* We can also draw more samples and take the mean of their gradients to reduce variation directly; as the mean of mostly uncorrelated vectors retains only the correlated component, which would be the true signal.

In the classical theory, under the classical assumptions of McCandish in the GNS paradigm, these are roughly equivalent modes of operation, and the industry has largely settled on the former. Optimizers are fed gradients at each step, and the noise level is low enough it largely does not matter.

## Empirical description

A wide range of probes were performed on anomalously productive training controllers that drew additional samples, modifying the underlying logical batch size using gradient accumulations. This was designed to autotune batch size, but did far more. Results were significant enough for a 50m test model to outperform an 800m control under isotoken budgets, and the control/test 50m case to outperform by 40% perplexity gains. This replicated across datasets and architectures.

# Implications.

## Optimizer theory

Optimizer theory based on GNS is not suitable for LLM training with AdamW using the standard training formula at small model sizes.

* The noise is provably incredibly high. Models that have gradient denoising take between 1/3rd to 1/10th the optimizer steps, suggesting the optimizers are mostly denoising
* Adam optimizers do not handle noise well. They slow down their training due to the noise, and respond aggressively to sudden changes in gradient magnitude due to the squared second moment. 

In summary, **Adam based optimizers perform poorly with noise**. The underlying mechanism conjectured to be the reason for this is 

* Adam maintains a running squared second moment that has been demeaned.
* This means a one-off, high magnitude second moment from a single batch disproportionally slows down training.

Other optimizers, particularly approximate second-order optimizers such as shampoo or K-FAc, likely suffer from this to an even greater degree. In contrast, if the step size of th3e gradients is constantly about the same norm, there is not quite the same opportunity for a particular secont moment to runaway and slow down training.

To summarize, the important conclusions are:

* **Adam's adaptive moments actively harm performance under high noise.** The squared second moment
* **Maintaining a constant gradient norm per optimizer step largely removes many of these issues**
* **It is worth including gradient accumulation mechanisms that estimate when the SNR is past a certain point and then step the optimizer in the LLM field.**
* **Gradient preconditioning to achieve a given Signal to Noise level is likely worthwhile**

## Controller theory

The implementation of a GQC-AS algorithm falls under the domain of a Sequential Binary Decision Controller. Gradients are drawn and averaged together in the torch accumulator, and the question asked each step is whether the quality is high enough it is worth stepping. If not, the controller draws another batch; else it steps the optimizer. Both mechanisms are reactive, and thus reach a lower floor than simply consistently rescaling the gradients to the same statically set magnitude. It is worth the extra hassle of gradient accumulation.

## Research Implications.

Several important details have emerged. 

* Gradient quality appears to matter a lot more than classically perceived. Some of the controller theory, notably the Gradient Noise Scale, does not appear to take into account the response of adam second moments to noise and appears to mispredict the optimum moment to step.
* If you directly control the magnitude of the gradients you do not also have to decay the learning rate, as you are ensuring smaller steps are taken at a higher quality to begin with.
* Denoising the gradients before they reach the optimizers with more microbatch samples appears to be a valid research direction.

Notably, a really thorough hyperparameter search was not possible at this time, as the experiments were run on a compute budget of around $150, but we attempted to follow de-facto scaling laws to compensate. Equal partners with more compute are welcome, and will be properly accredited. 

## Accessibility and Democratization

The power laws appear significantly better than standard laws.

* If this effect scales, it is likely GQC-AS controllers such as this will become a default part of most training formulations. Additionally, noise-sensitive models and optimizers may deserve a revisit.
* Regularization concerns appear when the model is not being trained on fresh data; turn up the regularization if using this when fine tuning!
* It may become viable for much smaller labs to train foundation models.

This is all contingent on the process scaling well.
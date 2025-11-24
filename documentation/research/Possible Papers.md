# Overview

This is a brief repository of certain, possible, or plausable papers that are intended

## Certain

These are the bare minimums, and I can execute this without additional funding in under 150$. However, better choicines would be preferred. 

### A Free Lunch: Adaptive Gradient Accumulation Universally Improves Performance and Wall Time

Discuss whatever relevant performance data we have over various models and datasets at around 120m. Discuss a control /test pair trained at 400m or so as well.

### A Gradient Cleaner Threshold Study for GPT2 at 120m

Find the best hyperparameters for 120m, using a small grid search or optuna script. Use pruning. Important for next step.

### GNTR Schedule Studies

Is a linear schedule, a cosine schedule, etc, best?

### Token Efficiency of Gradient Cleaning Variations: A Scaling Study

Kaplan scaling law fit at various token numbers. Probably around 100m, 200m, 400m, 800m tokens or so. Use best threshold parameters from last study, and for the cheap case only test control vs GNTS with no hyperparameter tuning.

## Plausible

Contingient on getting more funding or certain lines of research bearing out.

### Scaling studies on Gradient Cleaners

Unforunately, I think I need to find the relevant hyperparameter scaling laws. Quite a bit more expensive, in the 10,000-40,000$ range in theory. A less effective version can be done for under 10000, but may miss cases from the aggressive optuna pruning. Either way, still control vs GNTS unless something changes drastically, but we tune hyperparameters as well. This makes it comparable to Kaplan or Chinchilla. Train models at 50m, 100m, 200m, 400m, 800m, 1.6b parameters and tokens (grid) and do a power law fit with tokens and parameters.

### Gradient Cleaners Are Synergistic with Second Order Methods: Results for Shampoo and K-FAC

Given the amount of noise I now know is in the gradients, I have a sneaking suspicion second order methods have never actually gotten a chance to measure curvature. If so, adding gradient cleaners may significantly improve their performance.

### Gradient Cleaners For


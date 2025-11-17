"""
Metric-based hypothesis test controller for adaptive gradient accumulation.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from scipy import stats

from .base import AbstractOptimizerWrapper


class OptimizerWrapperMHT(AbstractOptimizerWrapper):
    """
    Metric Hypothesis Test (MHT) Controller

    This wrapper performs adaptive gradient accumulation based on statistical
    confidence in a metric estimate (typically loss). MHT accumulates batches
    until we are sufficiently confident (via t-test) that we have pinned down
    the true mean metric to within a specified error tolerance.

    The controller maintains a running EMA of the metric across steps and combines
    it with current-stage metrics to perform hypothesis testing at each batch.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The underlying optimizer to wrap
    confidence : float, optional (default: 0.98)
        Confidence level for hypothesis test (e.g., 0.98 = 98% confidence)
    error_tolerance : float, optional (default: 0.03)
        Maximum acceptable relative error in metric estimate (e.g., 0.03 = 3%)
    ema_alpha : float, optional (default: 0.01)
        EMA smoothing factor for running metric average
    max_batch_draws : int, optional (default: 64)
        Maximum accumulation steps before forcing an optimizer step
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        confidence: float = 0.98,
        error_tolerance: float = 0.03,
        ema_alpha: float = 0.01,
        max_batch_draws: int = 64,
    ):
        super().__init__(optimizer)

        self.confidence = confidence
        self.error_tolerance = error_tolerance
        self.ema_alpha = ema_alpha
        self.max_draws = max_batch_draws

        # State tracking
        self.running_avg_metric: Optional[float] = None
        self.current_stage_metrics = []

        # Note: self.parameters is constructed in base class.

    def step(
        self,
        metric: float,
        closure: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """
        Conditionally step the optimizer based on statistical confidence in metric estimate.

        Accumulates batches until the confidence interval of the mean metric
        fits within the specified error tolerance, then steps the optimizer.
        Call this once per batch, passing in the loss or other metric value.

        Parameters
        ----------
        metric : float
            The metric value from the current batch (typically loss)
        closure : callable, optional
            Optional closure for optimizers like LBFGS. If provided and it
            returns a value, the result will be cached in `last_optimizer_result`.

        Returns
        -------
        bool
            True if the optimizer stepped, False if still accumulating.
        """
        # Add metric to current stage
        self.current_stage_metrics.append(metric)

        # Setup running average if needed.
        if self.running_avg_metric is None:
            self.running_avg_metric = metric

        # Check if we should step
        force_step = self.num_draws >= self.max_draws
        alternative_hypothesis_accepted = self._is_null_hypothesis_rejected(
            self.current_stage_metrics,
            self.running_avg_metric,
            self.confidence,
            self.error_tolerance,
        )
        will_step_optimizer = alternative_hypothesis_accepted or force_step

        # Handle optimizer steps.
        if will_step_optimizer:
            # Step the optimizer
            self._take_optimizer_step(closure)

            # Update running average with mean of current stage
            stage_mean = np.mean(self.current_stage_metrics)
            self.running_avg_metric = (
                self.ema_alpha * stage_mean + (1 - self.ema_alpha) * self.running_avg_metric
            )

            # Reset accumulation state
            self.current_stage_metrics.clear()
        self._take_batch_step()
        return will_step_optimizer

    @staticmethod
    def _is_null_hypothesis_rejected(
        metrics: List[float],
        running_avg: float,
        confidence: float,
        error_tolerance: float,
    ) -> bool:
        """
        Test if confidence interval fits within error tolerance band.

        We test the null hypothesis: "Our estimate is not precise enough."
        We reject (return True) when the confidence interval fits within
        [mean * (1 - error_tolerance), mean * (1 + error_tolerance)]

        Parameters
        ----------
        metrics : List[float]
            Current stage metric values
        running_avg : float
            Running average from previous steps
        confidence : float
            Confidence level for the test (e.g., 0.98)
        error_tolerance : float
            Maximum acceptable relative error (e.g., 0.03)

        Returns
        -------
        bool
            True if null hypothesis is rejected (CI is tight enough to step)
        """
        if len(metrics) < 2:
            return False  # Need at least 2 samples for t-test

        # Combine running average with current stage then find the mean
        test_samples = np.array([running_avg] + metrics)
        mean = np.mean(test_samples)
        if mean == 0:
            return False  # Probably not enough draws.

        # Calculate confidence interval
        ci_low, ci_high = stats.t.interval(
            confidence=confidence,
            df=len(test_samples) - 1,
            loc=mean,
            scale=stats.sem(test_samples) + 1e-12,
        )

        # Check if CI fits within tolerance band
        tolerance_low = mean * (1 - error_tolerance)
        tolerance_high = mean * (1 + error_tolerance)

        # Reject null hypothesis if CI is contained within tolerance band
        return ci_low >= tolerance_low and ci_high <= tolerance_high

    def zero_grad(self):
        """Not needed - use step() method with metric parameter."""
        raise NotImplementedError(
            "MHT manages gradients internally. " "Call step(metric=...) instead of zero_grad()."
        )

    def statistics(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        statistics = self._get_base_statistics()
        statistics["running_avg_metric"] = self.running_avg_metric
        return statistics

    def __repr__(self):
        return (
            f"<OptimizerWrapperMHT("
            f"confidence={self.confidence:.2f}, "
            f"error_tol={self.error_tolerance:.2f}, "
            f"steps={self.num_steps}, "
            f"core={type(self.optimizer).__name__})>"
        )

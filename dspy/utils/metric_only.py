"""Metric-only mode for DSPy optimizers.

This module provides utilities to run optimizers without calling program.forward(),
only using the metric for scoring. This is useful for metrics that don't need
program generation, like TFLL which scores based on teacher-forced labels.
"""

from contextlib import contextmanager
import dspy


class NullLM:
    """Null LM that returns empty responses without making any API calls."""
    
    def chat(self, *args, **kwargs):
        """Return minimal chat response."""
        return {"choices": [{"message": {"content": ""}}]}
    
    def complete(self, *args, **kwargs):
        """Return minimal completion response."""
        return {"choices": [{"text": ""}]}
    
    def __call__(self, *args, **kwargs):
        """Default to chat mode."""
        return self.chat(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward method for DSPy compatibility."""
        return self.chat(*args, **kwargs)


@contextmanager
def metric_only_mode():
    """
    Context manager that temporarily replaces the LM with a NullLM.
    
    Usage:
        with metric_only_mode():
            best = optimizer.optimize(program, trainset, metric)
    
    This prevents any actual LM calls during optimization, allowing
    metrics like TFLL to work without generation costs.
    """
    prev_lm = dspy.settings.lm
    dspy.settings.configure(lm=NullLM())
    try:
        yield
    finally:
        dspy.settings.configure(lm=prev_lm)
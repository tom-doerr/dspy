"""Benchmark tests for TFLL RL optimizer."""

import time
import dspy
from dspy.teleprompt import TFLLRLOptimizer
from dspy.metrics.tfll import TFLLMetric


def benchmark_optimization():
    """Compare TFLL RL against baseline."""
    
    # Setup program
    class QA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict("q->a")
            
    prog = QA()
    
    # Real training data
    train = [
        dspy.Example(q="What is 2+2?", a="4"),
        dspy.Example(q="Capital of France?", a="Paris"),
    ]
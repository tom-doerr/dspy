#!/usr/bin/env python3
"""Visual test of TFLL RL optimizer."""

import logging
import time
import numpy as np
import dspy
from dspy.teleprompt.tfll_rl import TFLLRLOptimizer

logging.basicConfig(level=logging.INFO, format='%(message)s')

TARGET = 50

class VisualMetric:
    def __init__(self, target=TARGET):
        self.target = target
        self.calls = 0
        
    def __call__(self, example, program):
        self.calls += 1
        base = 45
        out = base + np.random.randint(-3, 3)
        reward = -abs(out - self.target)
        return reward


def main():
    print("\n" + "="*70)
    print("TFLL RL OPTIMIZER - VISUAL TEST")
    print("="*70)
    
    # Create training data
    trainset = []
    for i in range(5):
        ts = f"{time.time():.6f}"
        trainset.append(dspy.Example(timestamp=ts, output="x"*TARGET))
    
    # Create program
    class TestProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.signature = dspy.Signature("timestamp -> output")
            
        def forward(self, timestamp):
            return {"output": "x" * 30}
    
    program = TestProgram()
    print(f"Initial: '{program.signature.instructions}'")
    
    # Optimize
    metric = VisualMetric()
    optimizer = TFLLRLOptimizer(metric=metric, num_updates=3)
    optimized = optimizer.compile(program, trainset=trainset)
    
    print(f"Final: '{optimized.signature.instructions}'")
    print(f"Calls: {metric.calls}")


if __name__ == "__main__":
    main()
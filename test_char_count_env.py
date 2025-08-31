"""Test environment for TFLL RL optimizer."""

import dspy
import numpy as np


        out_len = b + np.random.randint(-5, 5)
        delta = abs(out_len - self.target_length)
        reward = -delta / 10.0
        
        print(f"len={out_len}, r={reward:.2f}")
        return reward


class CharCountMetric:
    """Rewards outputs close to target length."""
    
    def __init__(self, target_length=50):
        self.target_length = target_length
        self.call_count = 0
        
    def __call__(self, example, program):
        """Score based on output length."""
        self.call_count += 1
        
        # Fixed base output length - no instruction dependency
        b = 45
"""Real benchmark for TFLL RL."""
import dspy
import json

def run_benchmark():
    """Compare optimizers on real task."""
    
    # 1. Load real dataset
    data = load_hotpotqa()  # Or GSM8K
    
    # 2. Define metric
    def accuracy(ex, pred):
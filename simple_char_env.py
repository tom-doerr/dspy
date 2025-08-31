"""Simple env: output 50 chars."""
import time
import numpy as np
import argparse
import logging

TARGET = 50

class SimpleMetric:
    def __init__(self):
        self.calls = 0
        
    def __call__(self, ex, prog):
        self.calls += 1
        
        # Fixed base length - no instruction dependency
        b = 45
        
        out = b + np.random.randint(-3, 3)
        r = -abs(out - TARGET)
        
        print(f"[{self.calls}] len={out}, r={r}")
        return r


def test_with_timestamps(verbose=False, num_updates=20, show_search_replace=False):
    """Test using timestamps as unique inputs."""
    import time
    from dspy.teleprompt.tfll_rl import TFLLRLOptimizer
    import dspy
    
    # Create metric
    metric = SimpleMetric()
    
    # Create simple program
    class SimpleProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.signature = dspy.Signature("timestamp -> output")
            
        def forward(self, timestamp):
            # Dummy forward - metric only cares about instructions
            return {"output": "x" * 30}
    
    # Create training data with timestamps
    trainset = []
    for i in range(10):
        ts = f"{time.time():.6f}"
        trainset.append(dspy.Example(timestamp=ts, output="x"*50))
        time.sleep(0.001)  # Ensure unique timestamps
    
    # Create optimizer
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    optimizer = TFLLRLOptimizer(
        metric=metric,
        episodes_per_update=1,
        num_updates=num_updates,
        use_experience_replay=True
    )
    
    # Optimize with visualization
    program = SimpleProgram()
    
    if verbose or show_search_replace:
        print("\n" + "="*60)
        print("TFLL RL OPTIMIZATION PROCESS")
        print("="*60)
        print(f"Target: {TARGET} characters")
        print(f"Updates: {num_updates}")
        print(f"Initial prompt: '{program.signature.instructions}'")
        print("="*60 + "\n")
    
    optimized = optimizer.compile(program, trainset=trainset)
    
    if verbose or show_search_replace:
        print("\n" + "="*60)
        print(f"Final prompt: '{optimized.signature.instructions}'")
        print(f"Total metric calls: {metric.calls}")
        print("="*60)
    else:
        print(f"Final metric calls: {metric.calls}")
    
    return optimized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TFLL RL optimizer")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Show detailed output")
    parser.add_argument("-n", "--num-updates", type=int, default=20,
                        help="Number of optimization updates")
    parser.add_argument("-s", "--show-search-replace", action="store_true",
                        help="Show search/replace blocks")
    
    args = parser.parse_args()
    
    test_with_timestamps(
        verbose=args.verbose,
        num_updates=args.num_updates,
        show_search_replace=args.show_search_replace
    )
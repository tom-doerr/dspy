"""Simple env: output 50 chars."""
import time
import numpy as np

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


def test_with_timestamps():
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
    optimizer = TFLLRLOptimizer(
        metric=metric,
        episodes_per_update=1,
        num_updates=20,
        use_experience_replay=True
    )
    
    # Optimize
    program = SimpleProgram()
    optimized = optimizer.compile(program, trainset=trainset)
    
    print(f"Final metric calls: {metric.calls}")
    return optimized


if __name__ == "__main__":
    test_with_timestamps()
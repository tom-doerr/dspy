import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import SIMBAT
from dspy.utils.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def simple_metric(example, prediction, trace=None):
    """Simple metric that returns 1.0 for correct predictions, 0.0 otherwise."""
    if prediction is None:
        return 0.0
    return 1.0 if example.output == prediction.output else 0.0


def test_tail_eval_simba_initialization():
    """Test that TailEvalSIMBA initializes correctly."""
    optimizer = SIMBAT(
        metric=simple_metric,
        bsize=4,
        num_candidates=2,
        max_steps=1,
        max_demos=2
    )
    assert optimizer.metric == simple_metric
    assert optimizer.bsize == 4
    assert optimizer.num_candidates == 2
    assert optimizer.max_steps == 1
    assert optimizer.max_demos == 2


def test_tail_eval_simba_compile_basic():
    """Test basic compilation with TailEvalSIMBA."""
    # Create a simple trainset
    trainset = [
        Example(input="What is 2+2?", output="4").with_inputs("input"),
        Example(input="What is 3+3?", output="6").with_inputs("input"),
        Example(input="What is 4+4?", output="8").with_inputs("input"),
        Example(input="What is 5+5?", output="10").with_inputs("input"),
        Example(input="What is 6+6?", output="12").with_inputs("input"),
        Example(input="What is 7+7?", output="14").with_inputs("input"),
        Example(input="What is 8+8?", output="16").with_inputs("input"),
        Example(input="What is 9+9?", output="18").with_inputs("input"),
    ]
    
    # Configure with dummy LM
    lm = DummyLM(["4", "6", "8", "10", "12", "14", "16", "18"])
    dspy.settings.configure(lm=lm)
    
    # Create student module
    student = SimpleModule("input -> output")
    
    # Create optimizer with small parameters for testing
    optimizer = SIMBAT(
        metric=simple_metric,
        bsize=2,
        num_candidates=2,
        max_steps=1,
        max_demos=1,
        num_threads=1
    )
    
    # Compile with tail evaluation
    compiled_student = optimizer.compile(
        student=student,
        trainset=trainset,
        seed=42,
        tail_eval_n=4,
        dedup_seen=True
    )
    
    # Check that compilation succeeded and returned a module
    assert compiled_student is not None
    assert hasattr(compiled_student, 'candidate_programs')
    assert hasattr(compiled_student, 'trial_logs')


def test_tail_eval_simba_dedup_seen():
    """Test that dedup_seen parameter works correctly."""
    # Create a larger trainset
    trainset = [
        Example(input=f"What is {i}+{i}?", output=str(2*i)).with_inputs("input")
        for i in range(10)
    ]
    
    # Configure with dummy LM
    lm = DummyLM([str(2*i) for i in range(10)])
    dspy.settings.configure(lm=lm)
    
    student = SimpleModule("input -> output")
    
    # Test with dedup_seen=True
    optimizer = SIMBAT(
        metric=simple_metric,
        bsize=3,
        num_candidates=2,
        max_steps=2,
        max_demos=0,
        num_threads=1
    )
    
    compiled_with_dedup = optimizer.compile(
        student=student,
        trainset=trainset,
        seed=42,
        tail_eval_n=5,
        dedup_seen=True
    )
    
    # Test with dedup_seen=False
    compiled_without_dedup = optimizer.compile(
        student=student,
        trainset=trainset,
        seed=42,
        tail_eval_n=5,
        dedup_seen=False
    )
    
    # Both should compile successfully
    assert compiled_with_dedup is not None
    assert compiled_without_dedup is not None


def test_tail_eval_simba_small_trainset():
    """Test that TailEvalSIMBA handles small trainsets correctly."""
    # Create a minimal trainset
    trainset = [
        Example(input="What is 1+1?", output="2").with_inputs("input"),
        Example(input="What is 2+2?", output="4").with_inputs("input"),
    ]
    
    lm = DummyLM(["2", "4"])
    dspy.settings.configure(lm=lm)
    
    student = SimpleModule("input -> output")
    
    optimizer = SIMBAT(
        metric=simple_metric,
        bsize=2,  # Same size as trainset
        num_candidates=1,
        max_steps=1,
        max_demos=0
    )
    
    # Should still work even when tail_eval_n is larger than trainset
    compiled_student = optimizer.compile(
        student=student,
        trainset=trainset,
        seed=42,
        tail_eval_n=10,  # Larger than trainset
        dedup_seen=True
    )
    
    assert compiled_student is not None


def test_tail_eval_simba_tail_size_limits():
    """Test that tail_eval_n is properly limited to trainset size."""
    trainset = [
        Example(input=f"What is {i}?", output=str(i)).with_inputs("input")
        for i in range(5)
    ]
    
    lm = DummyLM([str(i) for i in range(5)])
    dspy.settings.configure(lm=lm)
    
    student = SimpleModule("input -> output")
    
    optimizer = SIMBAT(
        metric=simple_metric,
        bsize=2,
        num_candidates=1,
        max_steps=1
    )
    
    # Request more tail examples than available
    compiled_student = optimizer.compile(
        student=student,
        trainset=trainset,
        seed=42,
        tail_eval_n=100,  # Much larger than trainset
        dedup_seen=True
    )
    
    # Should compile successfully with limited tail size
    assert compiled_student is not None
    assert hasattr(compiled_student, 'trial_logs')


def test_tail_eval_simba_invalid_trainset():
    """Test that TailEvalSIMBA raises appropriate error for too small trainset."""
    trainset = [Example(input="What is 1?", output="1").with_inputs("input")]
    
    lm = DummyLM(["1"])
    dspy.settings.configure(lm=lm)
    
    student = SimpleModule("input -> output")
    
    optimizer = SIMBAT(
        metric=simple_metric,
        bsize=2,  # Larger than trainset
        num_candidates=1,
        max_steps=1
    )
    
    with pytest.raises(AssertionError, match="Trainset too small"):
        optimizer.compile(
            student=student,
            trainset=trainset,
            seed=42,
            tail_eval_n=1
        )


def test_simbat_parallel_reflection():
    """Test SIMBAT runs strategies in parallel with multiple threads."""
    trainset = [
        Example(input=f"Q{i}", output=f"A{i}").with_inputs("input")
        for i in range(8)
    ]
    lm = DummyLM([f"A{i}" for i in range(8)])
    dspy.settings.configure(lm=lm)

    student = SimpleModule("input -> output")
    optimizer = SIMBAT(
        metric=simple_metric, bsize=4, num_candidates=3,
        max_steps=1, max_demos=1, num_threads=4
    )
    compiled = optimizer.compile(student=student, trainset=trainset, seed=42)
    assert compiled is not None
    assert hasattr(compiled, 'candidate_programs')


def test_simbat_parallel_with_demos():
    """Test parallel reflection with append_a_demo strategy."""
    trainset = [
        Example(input=f"Q{i}", output=f"A{i}").with_inputs("input")
        for i in range(8)
    ]
    lm = DummyLM([f"A{i}" for i in range(8)])
    dspy.settings.configure(lm=lm)
    student = SimpleModule("input -> output")
    optimizer = SIMBAT(
        metric=simple_metric, bsize=4, num_candidates=2,
        max_steps=2, max_demos=2, num_threads=2
    )
    compiled = optimizer.compile(student=student, trainset=trainset, seed=42)
    assert compiled is not None


def test_simbat_single_thread():
    """Test SIMBAT works with num_threads=1."""
    trainset = [
        Example(input=f"Q{i}", output=f"A{i}").with_inputs("input")
        for i in range(6)
    ]
    lm = DummyLM([f"A{i}" for i in range(6)])
    dspy.settings.configure(lm=lm)
    student = SimpleModule("input -> output")
    optimizer = SIMBAT(
        metric=simple_metric, bsize=3, num_candidates=2,
        max_steps=1, max_demos=0, num_threads=1
    )
    compiled = optimizer.compile(student=student, trainset=trainset, seed=42)
    assert compiled is not None


def test_simbat_multiple_batches():
    """Test SIMBAT with multiple batches and high parallelism."""
    trainset = [Example(input=f"Q{i}", output=f"A{i}").with_inputs("input")
        for i in range(16)]
    lm = DummyLM([f"A{i}" for i in range(16)])
    dspy.settings.configure(lm=lm)
    student = SimpleModule("input -> output")
    optimizer = SIMBAT(
        metric=simple_metric, bsize=4, num_candidates=4,
        max_steps=3, max_demos=1, num_threads=8
    )
    compiled = optimizer.compile(student=student, trainset=trainset, seed=42)
    assert compiled is not None
    assert hasattr(compiled, 'trial_logs')
    assert len(compiled.trial_logs) == 3  # 3 batches
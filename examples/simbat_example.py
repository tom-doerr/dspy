"""
Example demonstrating the use of SIMBAT optimizer.

SIMBAT (SIMBA with Tail evaluation) is a variant of SIMBA that evaluates
on a "tail" subset of the training data instead of the full trainset
during final validation.
"""

import dspy
from dspy import Example


# Define a simple QA signature
class BasicQA(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


# Create a simple module
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, question):
        return self.generate_answer(question=question)


# Define a metric
def qa_metric(example, prediction, trace=None):
    """Simple metric that checks if the answer is correct."""
    if prediction is None:
        return 0.0
    # In a real scenario, you'd have more sophisticated matching
    return 1.0 if example.answer.lower() in prediction.answer.lower() else 0.0


# Create training data
trainset = [
    Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    Example(question="What is 2+2?", answer="4").with_inputs("question"),
    Example(question="What color is the sky?", answer="blue").with_inputs("question"),
    Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare").with_inputs("question"),
    Example(question="What is the largest planet?", answer="Jupiter").with_inputs("question"),
    Example(question="What is water made of?", answer="H2O").with_inputs("question"),
    Example(question="How many continents are there?", answer="7").with_inputs("question"),
    Example(question="What is the speed of light?", answer="299792458 m/s").with_inputs("question"),
]


def main():
    # Configure DSPy (in practice, you'd use a real LM)
    # lm = dspy.OpenAI(model="gpt-3.5-turbo")
    # dspy.settings.configure(lm=lm)
    
    # For this example, we'll use a dummy LM
    from dspy.utils.dummies import DummyLM
    lm = DummyLM(["Paris", "4", "blue", "Shakespeare", "Jupiter", "H2O", "7", "299792458 m/s"])
    dspy.settings.configure(lm=lm)
    
    # Create the student module
    student = SimpleQA()
    
    # Create SIMBAT optimizer
    optimizer = dspy.SIMBAT(
        metric=qa_metric,
        bsize=2,           # Mini-batch size
        num_candidates=3,  # Number of candidate programs per iteration
        max_steps=2,       # Number of optimization steps
        max_demos=2,       # Maximum demos per predictor
        tail_eval_n=4,     # Size of tail evaluation set
    )
    
    # Compile the student
    print("Compiling with SIMBAT...")
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        seed=42,
        tail_eval_n=4,     # Evaluate on 4 examples from the tail
        dedup_seen=True    # Skip examples seen during optimization
    )
    
    print("Compilation complete!")
    
    # Test the optimized module
    test_question = "What is the capital of France?"
    result = optimized(question=test_question)
    print(f"\nTest Question: {test_question}")
    print(f"Answer: {result.answer}")
    
    # Show candidate programs and their scores
    if hasattr(optimized, 'candidate_programs'):
        print(f"\nFound {len(optimized.candidate_programs)} candidate programs")
        for i, candidate in enumerate(optimized.candidate_programs):
            print(f"  Candidate {i+1}: score = {candidate['score']:.3f}")


if __name__ == "__main__":
    main()
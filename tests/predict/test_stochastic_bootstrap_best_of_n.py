import dspy
from dspy.predict.stochastic_bootstrap_best_of_n import StochasticBootstrapBestOfN
from dspy.utils.dummies import DummyLM


def test_signature_constructor_and_threshold():
    lm = DummyLM([{"answer": "Ghent"}, {"answer": "Brussels"}])
    dspy.settings.configure(lm=lm)

    def reward_fn(_, pred):
        return 1.0 if getattr(pred, "answer", "") == "Brussels" else 0.0

    chooser = StochasticBootstrapBestOfN(
        module_or_signature="question -> answer",
        reward_fn=reward_fn,
        N=2,
        threshold=1.0,
        k=0,
        max_bootstrapped_demos=0,
        num_workers=1,
    )

    result = chooser(question="What is the capital of Belgium?")
    assert result.answer == "Brussels"

class DemoAwareModule(dspy.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question, config=None):
        for demo in getattr(self.predictor, "demos", []):
            value = getattr(demo, "question", None)
            if value is None and isinstance(demo, dict):
                value = demo.get("question")
            if value == question:
                answer = getattr(demo, "answer", None)
                if answer is None and isinstance(demo, dict):
                    answer = demo.get("answer")
                return dspy.Example(question=question, answer=answer)
        rid = (config or {}).get("rollout_id", 0)
        answer = self.outputs[rid % len(self.outputs)]
        return dspy.Example(question=question, answer=answer)

def test_demo_bootstrapping_and_reuse():
    module = DemoAwareModule(["wrong", "blue", "wrong"])

    def reward_fn(kwargs, pred):
        gold = kwargs["question"]
        return 1.0 if getattr(pred, "answer", "") == "blue" and "sky" in gold else 0.0

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=reward_fn,
        N=3,
        threshold=1.0,
        k=1,
        max_bootstrapped_demos=4,
        num_workers=1,
    )

    first = chooser(question="What color is the sky?")
    assert first.answer == "blue"

    module.outputs = ["wrong"]
    second = chooser(question="What color is the sky?")
    assert second.answer == "blue"

    memory_key = chooser._memory_key({"question": "What color is the sky?"})
    records = chooser._memory[memory_key]
    assert records
    assert any(record.example.answer == "blue" for record in records)

import pytest

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
        max_bootstrapped_demos=0,
        replay_buffer_size=0,
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
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))

    def reward_fn(kwargs, pred):
        gold = kwargs["question"]
        return 1.0 if getattr(pred, "answer", "") == "blue" and "sky" in gold else 0.0

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=reward_fn,
        N=3,
        threshold=1.0,
        max_bootstrapped_demos=1,
        replay_buffer_size=4,
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


def test_memory_snapshot_and_reset():
    module = DemoAwareModule(["blue"])
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))

    def reward_fn(kwargs, pred):
        gold = kwargs["question"]
        return 1.0 if getattr(pred, "answer", "") == "blue" and "sky" in gold else 0.0

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=reward_fn,
        N=1,
        threshold=0.9,
        max_bootstrapped_demos=1,
        replay_buffer_size=2,
        num_workers=1,
    )

    chooser(question="What color is the sky?")
    snapshot = chooser.memory_snapshot()
    assert snapshot
    key = next(iter(snapshot))
    assert len(snapshot[key]) == 1

    chooser.reset_memory(key)
    assert not chooser.memory_snapshot()


class LoggingModule(dspy.Module):
    logs: list[list[str]] = []

    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question, config=None):
        slate = []
        for demo in getattr(self.predictor, "demos", []):
            if isinstance(demo, dspy.Example):
                slate.append(getattr(demo, "answer", None))
            elif isinstance(demo, dict):
                slate.append(demo.get("answer"))
        LoggingModule.logs.append(slate)
        rid = (config or {}).get("rollout_id", 0)
        answer = self.outputs[rid % len(self.outputs)]
        return dspy.Example(question=question, answer=answer).with_inputs("question")


def test_candidate_demo_count_matches_cap():
    LoggingModule.logs = []
    module = LoggingModule(["blue", "blue", "blue"])
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: 1.0,
        N=3,
        max_bootstrapped_demos=2,
        replay_buffer_size=4,
        seed=0,
        num_workers=1,
    )

    chooser(question="warmup one")
    LoggingModule.logs.clear()
    chooser(question="warmup two")
    LoggingModule.logs.clear()

    chooser(question="final run")
    assert len(LoggingModule.logs) == 3
    assert all(len(slate) == 2 for slate in LoggingModule.logs)


class UniqueAnswerModule(dspy.Module):
    counter = 0

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question, config=None):
        UniqueAnswerModule.counter += 1
        answer = f"ans{UniqueAnswerModule.counter}"
        return dspy.Example(question=question, answer=answer).with_inputs("question")


def test_replay_buffer_respects_capacity():
    UniqueAnswerModule.counter = 0
    module = UniqueAnswerModule()
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: 1.0,
        N=1,
        max_bootstrapped_demos=1,
        replay_buffer_size=2,
        num_workers=1,
    )

    for _ in range(3):
        chooser(question="repeat question")

    snapshot = chooser.memory_snapshot()
    assert snapshot
    key = next(iter(snapshot))
    assert len(snapshot[key]) == 2


class ThresholdModule(dspy.Module):
    score_queue: list[float] = []
    call_count: int = 0

    def __init__(self):
        super().__init__()

    def forward(self, question, config=None):
        ThresholdModule.call_count += 1
        score = ThresholdModule.score_queue.pop(0)
        return dspy.Example(question=question, answer="dummy", score=score).with_inputs("question")


def test_threshold_stops_early():
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))
    ThresholdModule.score_queue = [0.9, 0.1, 0.2]
    ThresholdModule.call_count = 0
    module = ThresholdModule()

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: pred.score,
        N=3,
        threshold=0.8,
        max_bootstrapped_demos=0,
        replay_buffer_size=0,
        num_workers=1,
    )

    chooser(question="stop once high reward appears")
    assert ThresholdModule.call_count == 1


class FlakyModule(dspy.Module):
    failures_left: int = 0

    def __init__(self, failures_before_success):
        super().__init__()
        FlakyModule.failures_left = failures_before_success
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question, config=None):
        if FlakyModule.failures_left > 0:
            FlakyModule.failures_left -= 1
            raise RuntimeError("boom")
        return dspy.Example(question=question, answer="ok").with_inputs("question")


def test_fail_count_enforced():
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))
    module = FlakyModule(failures_before_success=2)
    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: 1.0,
        N=3,
        fail_count=1,
        max_bootstrapped_demos=0,
        replay_buffer_size=0,
        num_workers=1,
    )

    with pytest.raises(RuntimeError):
        chooser(question="trigger failure")


def test_fail_count_allows_recovery():
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))
    module = FlakyModule(failures_before_success=1)
    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: 1.0,
        N=2,
        fail_count=1,
        max_bootstrapped_demos=0,
        replay_buffer_size=0,
        num_workers=1,
    )

    result = chooser(question="recover after one failure")
    assert result.answer == "ok"


class ConfigAwareModule(dspy.Module):
    captured: list[dict[str, float]] = []

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer", temperature=0.3)

    def forward(self, question, config=None):
        ConfigAwareModule.captured.append(dict(self.predictor.config))
        return dspy.Example(question=question, answer="ok").with_inputs("question")


def test_predictor_config_restored():
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))
    module = ConfigAwareModule()
    ConfigAwareModule.captured = []

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: 1.0,
        N=1,
        max_bootstrapped_demos=0,
        replay_buffer_size=0,
        num_workers=1,
    )

    chooser(question="check config", config={"top_p": 0.6})
    assert ConfigAwareModule.captured and ConfigAwareModule.captured[0]["top_p"] == 0.6


class ParallelLoggingModule(dspy.Module):
    captured = []

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question, config=None):
        ParallelLoggingModule.captured.append(config["rollout_id"])
        return dspy.Example(question=question, answer="ok").with_inputs("question")


def test_parallel_execution_preserves_attempt_count():
    ParallelLoggingModule.captured = []
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))
    module = ParallelLoggingModule()

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: 0.1,
        N=5,
        max_bootstrapped_demos=0,
        replay_buffer_size=0,
        num_workers=4,
        seed=42,
    )

    chooser(question="parallel run")
    assert len(ParallelLoggingModule.captured) == 5
    assert len(set(ParallelLoggingModule.captured)) == 5


def test_recency_bonus_keeps_single_demo_selectable():
    module = DemoAwareModule(["blue"])
    dspy.settings.configure(lm=DummyLM([{"answer": "unused"}]))

    chooser = StochasticBootstrapBestOfN(
        module_or_signature=module,
        reward_fn=lambda kwargs, pred: 1.0,
        N=2,
        max_bootstrapped_demos=1,
        replay_buffer_size=1,
        recency_weight=1.5,
        recency_tau=1.0,
        num_workers=1,
    )

    for _ in range(3):
        chooser(question="What color is the sky?")

    snapshot = chooser.memory_snapshot()
    key = next(iter(snapshot))
    assert len(snapshot[key]) == 1

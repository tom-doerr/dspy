import json
import re

import litellm

import dspy
from dspy import Example
from dspy.dsp.utils.utils import dotdict
from dspy.teleprompt import Direct
from dspy.teleprompt.direct import (
    DirectEditStep,
    DirectHistoryEntry,
    SearchReplaceBlock,
    _apply_instruction_edits,
    _summarize_sample_for_history,
)
from dspy.utils.dummies import DummyLM


def score_match(example, prediction):
    if prediction is None:
        return 0.0
    return 1.0 if prediction.answer == example.answer else 0.0


def prediction_score_match(example, prediction):
    return dspy.Prediction(score=score_match(example, prediction))


class SimpleModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def _extract_field(message_content: str, field_name: str) -> str:
    pattern = rf"\[\[ ## {field_name} ## \]\]\n(.*?)(?=\n\n\[\[ ## |\Z)"
    match = re.search(pattern, message_content, re.DOTALL)
    assert match is not None, f"Missing field {field_name} in:\n{message_content}"
    value = match.group(1).strip()
    return value.split("\n\nRespond with", 1)[0].strip()


class RuleAwareTaskLM(DummyLM):
    def __init__(self, question_rules: dict[str, tuple[str, str]]):
        super().__init__([])
        self.question_rules = question_rules

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        system_prompt = messages[0]["content"]
        question = _extract_field(messages[-1]["content"], "question")
        required_rule, answer = self.question_rules[question]
        output = answer if required_rule in system_prompt else "wrong"
        return dotdict(
            choices=[dotdict(message=dotdict(content=self._format_answer_fields({"answer": output}), tool_calls=None), finish_reason="stop")],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="dummy",
        )


class InstructionAnswerTaskLM(DummyLM):
    def __init__(self, keyword_answers: list[tuple[str, str]], default_answer: str = "wrong"):
        super().__init__([])
        self.keyword_answers = keyword_answers
        self.default_answer = default_answer

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        system_prompt = messages[0]["content"]
        output = self.default_answer
        for keyword, answer in self.keyword_answers:
            if keyword in system_prompt:
                output = answer
                break
        return dotdict(
            choices=[dotdict(message=dotdict(content=self._format_answer_fields({"answer": output}), tool_calls=None), finish_reason="stop")],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="dummy",
        )


class HistoryAwareDirectLM(DummyLM):
    def __init__(self):
        super().__init__([])

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        history = json.loads(_extract_field(messages[-1]["content"], "history"))
        if len(history) > 1:
            raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")

        latest_question = history[-1]["sample"]["question"]
        replacement = {
            "q1": "If question is q1, answer blue.",
            "q2": "If question is q2, answer green.",
        }[latest_question]
        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(
                        content=self._format_answer_fields(
                            {"module_edits": {"predictor": [{"search": "", "replace": replacement}]}}
                        ),
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="dummy",
        )


class RecordingMetric:
    def __init__(self):
        self.calls = []
        self.initial_calls = []
        self.reflection_calls = []

    def __call__(self, example, prediction):
        self.calls.append(getattr(example, "question", None))
        return score_match(example, prediction)

    def record_initial_train_metric(self, example, prediction, score):
        self.initial_calls.append((getattr(example, "question", None), score))

    def record_reflection_metric(self, example, prediction, score):
        self.reflection_calls.append((getattr(example, "question", None), score))


def test_direct_module_choice_and_signature_order():
    optimizer = Direct(metric=score_match)
    assert isinstance(optimizer.optimizer_module, dspy.Predict)
    assert optimizer.max_iters_per_mistake == 3
    assert optimizer.min_metric_to_skip == 1.0
    assert list(optimizer.optimizer_module.signature.input_fields.keys()) == [
        "history",
        "current_instructions",
        "module_names",
    ]

    cot_optimizer = Direct(metric=score_match, use_cot=True)
    assert isinstance(cot_optimizer.optimizer_module, dspy.ChainOfThought)
    assert list(cot_optimizer.optimizer_module.predict.signature.input_fields.keys()) == [
        "history",
        "current_instructions",
        "module_names",
    ]


def test_apply_instruction_edits_respects_search_replace_semantics():
    updated, applied = _apply_instruction_edits(
        "base",
        [
            SearchReplaceBlock(search="", replace="append"),
            SearchReplaceBlock(search="base", replace="new"),
            SearchReplaceBlock(search="missing", replace="ignored"),
        ],
    )
    assert applied is True
    assert updated == "new\n\nappend"
    assert "ignored" not in updated

    deleted, applied = _apply_instruction_edits(
        "delete me",
        [SearchReplaceBlock(search=" me", replace="")],
    )
    assert applied is True
    assert deleted == "delete"


def test_direct_compile_fixes_single_mistake():
    task_lm = RuleAwareTaskLM({"q1": ("If question is q1, answer blue.", "blue")})
    optimizer_lm = DummyLM(
        [
            {
                "module_edits": {
                    "predictor": [
                        {"search": "", "replace": "If question is q1, answer blue."},
                    ]
                }
            }
        ]
    )

    student = SimpleModule()
    student.set_lm(task_lm)
    optimizer = Direct(metric=score_match, prompt_model=optimizer_lm)

    trainset = [Example(question="q1", answer="blue").with_inputs("question")]
    optimized = optimizer.compile(student, trainset=trainset)

    assert "If question is q1, answer blue." in optimized.predictor.signature.instructions
    assert optimized(question="q1").answer == "blue"
    assert optimized.direct_history[0]["edits"][0]["resulting_metric"] == 1.0


def test_direct_reverts_failed_edit_but_keeps_attempt_in_history():
    task_lm = RuleAwareTaskLM({"q1": ("If question is q1, answer blue.", "blue")})
    optimizer_lm = DummyLM(
        [
            {
                "module_edits": {
                    "predictor": [
                        {"search": "", "replace": "If question is q1, answer red."},
                    ]
                }
            },
            {
                "module_edits": {
                    "predictor": [
                        {"search": "", "replace": "If question is q1, answer blue."},
                    ]
                }
            },
        ]
    )

    student = SimpleModule()
    student.set_lm(task_lm)
    optimizer = Direct(metric=score_match, prompt_model=optimizer_lm, max_iters_per_mistake=2)

    trainset = [Example(question="q1", answer="blue").with_inputs("question")]
    optimized = optimizer.compile(student, trainset=trainset)

    assert optimized(question="q1").answer == "blue"
    assert "If question is q1, answer blue." in optimized.predictor.signature.instructions
    assert "If question is q1, answer red." not in optimized.predictor.signature.instructions
    assert [step["resulting_metric"] for step in optimized.direct_history[0]["edits"]] == [0.0, 1.0]
    assert [step["accepted"] for step in optimized.direct_history[0]["edits"]] == [False, True]
    assert optimized.direct_history[0]["edits"][0]["edits"]["predictor"][0]["replace"] == "If question is q1, answer red."
    assert optimized.direct_stats["edits_applied"] == 1


def test_direct_accepts_improving_edits_until_threshold():
    def score_answer(example, prediction):
        if prediction is None:
            return 0.0
        return {"wrong": 0.0, "mid": 0.4, "better": 0.7, "best": 1.0}[prediction.answer]

    task_lm = InstructionAnswerTaskLM(
        [
            ("best rule", "best"),
            ("better rule", "better"),
            ("mid rule", "mid"),
        ]
    )
    optimizer_lm = DummyLM(
        [
            {"module_edits": {"predictor": [{"search": "", "replace": "mid rule"}]}},
            {"module_edits": {"predictor": [{"search": "mid rule", "replace": "better rule"}]}},
            {"module_edits": {"predictor": [{"search": "better rule", "replace": "best rule"}]}},
        ]
    )

    student = SimpleModule()
    student.set_lm(task_lm)
    optimizer = Direct(metric=score_answer, prompt_model=optimizer_lm, max_iters_per_mistake=3)

    trainset = [Example(question="q1").with_inputs("question")]
    optimized = optimizer.compile(student, trainset=trainset)

    assert optimized(question="q1").answer == "best"
    assert "best rule" in optimized.predictor.signature.instructions
    assert "better rule" not in optimized.predictor.signature.instructions
    assert [step["resulting_metric"] for step in optimized.direct_history[0]["edits"]] == [0.4, 0.7, 1.0]
    assert [step["accepted"] for step in optimized.direct_history[0]["edits"]] == [True, True, True]
    assert optimized.direct_stats["samples_optimized"] == 1
    assert optimized.direct_stats["edits_applied"] == 3


def test_direct_skips_scores_at_or_above_default_threshold():
    def score_answer(example, prediction):
        if prediction is None:
            return 0.0
        return {"good": 1.5}.get(prediction.answer, 0.0)

    task_lm = InstructionAnswerTaskLM([], default_answer="good")
    student = SimpleModule()
    student.set_lm(task_lm)
    optimizer = Direct(metric=score_answer, prompt_model=DummyLM([]))

    trainset = [Example(question="q1").with_inputs("question")]
    optimized = optimizer.compile(student, trainset=trainset)

    assert optimized(question="q1").answer == "good"
    assert optimized.direct_history == []
    assert optimized.direct_stats["samples_skipped"] == 1
    assert optimized.direct_stats["samples_optimized"] == 0


def test_direct_can_always_optimize_when_skip_threshold_is_none():
    def score_answer(example, prediction):
        if prediction is None:
            return 0.0
        return {"good": 1.0, "great": 2.0}.get(prediction.answer, 0.0)

    task_lm = InstructionAnswerTaskLM([("great rule", "great")], default_answer="good")
    optimizer_lm = DummyLM(
        [
            {"module_edits": {"predictor": [{"search": "", "replace": "great rule"}]}},
        ]
    )

    student = SimpleModule()
    student.set_lm(task_lm)
    optimizer = Direct(metric=score_answer, prompt_model=optimizer_lm, min_metric_to_skip=None, max_iters_per_mistake=1)

    trainset = [Example(question="q1").with_inputs("question")]
    optimized = optimizer.compile(student, trainset=trainset)

    assert optimized(question="q1").answer == "great"
    assert optimized.direct_history[0]["metric"] == 1.0
    assert optimized.direct_history[0]["edits"][0]["resulting_metric"] == 2.0
    assert optimized.direct_history[0]["edits"][0]["accepted"] is True
    assert optimized.direct_stats["samples_skipped"] == 0
    assert optimized.direct_stats["samples_optimized"] == 1


def test_direct_halves_history_after_context_error():
    task_lm = RuleAwareTaskLM(
        {
            "q1": ("If question is q1, answer blue.", "blue"),
            "q2": ("If question is q2, answer green.", "green"),
        }
    )
    student = SimpleModule()
    student.set_lm(task_lm)

    optimizer = Direct(metric=score_match, prompt_model=HistoryAwareDirectLM())
    trainset = [
        Example(question="q1", answer="blue").with_inputs("question"),
        Example(question="q2", answer="green").with_inputs("question"),
    ]
    optimized = optimizer.compile(student, trainset=trainset)

    assert optimized(question="q1").answer == "blue"
    assert optimized(question="q2").answer == "green"
    assert optimized.direct_stats["context_retries"] == 1
    assert optimized.direct_stats["history_halvings"] == 1
    assert len(optimized.direct_history) == 1
    assert optimized.direct_history[0]["sample"]["question"] == "q2"


def test_direct_compile_runs_final_valset_eval():
    task_lm = RuleAwareTaskLM({"q1": ("If question is q1, answer blue.", "blue")})
    optimizer_lm = DummyLM(
        [
            {
                "module_edits": {
                    "predictor": [
                        {"search": "", "replace": "If question is q1, answer blue."},
                    ]
                }
            }
        ]
    )

    student = SimpleModule()
    student.set_lm(task_lm)
    optimizer = Direct(metric=prediction_score_match, prompt_model=optimizer_lm)

    trainset = [Example(question="q1", answer="blue").with_inputs("question")]
    valset = [Example(question="q1", answer="blue").with_inputs("question")]
    optimized = optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
        eval_kwargs={"display_progress": False},
    )

    assert optimized.direct_valset_result is not None
    assert optimized.direct_valset_score == 100.0
    assert optimized.direct_valset_result.results[0][2] == 1.0


def test_direct_records_initial_train_metric_once_per_train_example():
    task_lm = RuleAwareTaskLM({"q1": ("If question is q1, answer blue.", "blue")})
    optimizer_lm = DummyLM(
        [
            {
                "module_edits": {
                    "predictor": [
                        {"search": "", "replace": "If question is q1, answer blue."},
                    ]
                }
            }
        ]
    )
    metric = RecordingMetric()

    student = SimpleModule()
    student.set_lm(task_lm)
    optimizer = Direct(metric=metric, prompt_model=optimizer_lm)

    trainset = [Example(question="q1", answer="blue").with_inputs("question")]
    optimized = optimizer.compile(student, trainset=trainset)

    assert optimized(question="q1").answer == "blue"
    assert metric.calls == ["q1", "q1"]
    assert metric.initial_calls == [("q1", 0.0)]
    assert metric.reflection_calls == [("q1", 1.0)]


def test_direct_halves_single_history_entry_by_edit_history():
    optimizer = Direct(metric=score_match)
    history = [
        DirectHistoryEntry(
            sample={"question": "q"},
            metric=0.0,
            edits=[
                DirectEditStep(edits={}, resulting_metric=0.2, accepted=True),
                DirectEditStep(edits={}, resulting_metric=0.1, accepted=False),
                DirectEditStep(edits={}, resulting_metric=0.5, accepted=True),
                DirectEditStep(edits={}, resulting_metric=0.4, accepted=False),
            ],
        )
    ]

    assert optimizer._halve_history(history) is True
    assert history[0].metric == 0.2
    assert [step.resulting_metric for step in history[0].edits] == [0.5, 0.4]


def test_direct_history_summarizes_images_without_base64_payloads():
    sample = {
        "question": "q",
        "post_image": dspy.Image("https://example.com/test.png"),
    }
    summarized = _summarize_sample_for_history(sample)
    assert summarized["question"] == "q"
    assert summarized["post_image"].startswith("Image(")
    assert "https://example.com/test.png" in summarized["post_image"]

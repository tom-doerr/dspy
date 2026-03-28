from __future__ import annotations

import json
import logging
import numbers
from typing import Any

import litellm
import pydantic

import dspy
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)


class SearchReplaceBlock(pydantic.BaseModel):
    search: str = pydantic.Field(
        default="",
        description="Exact instruction text to search for. Empty search means append the replacement.",
    )
    replace: str = pydantic.Field(
        default="",
        description="Replacement instruction text. Empty replace deletes the searched text.",
    )


class DirectEditStep(pydantic.BaseModel):
    edits: dict[str, list[SearchReplaceBlock]] = pydantic.Field(
        default_factory=dict,
        description="Search/replace blocks that were applied to predictor instructions.",
    )
    resulting_metric: float = pydantic.Field(
        description="Metric value after applying the edits.",
    )


class DirectHistoryEntry(pydantic.BaseModel):
    sample: dict[str, Any] = pydantic.Field(
        description="The sample that produced a metric value at or below zero.",
    )
    metric: float = pydantic.Field(
        description="Metric value for the sample before any edits in this history entry.",
    )
    edits: list[DirectEditStep] = pydantic.Field(
        default_factory=list,
        description="All edits attempted for this sample, in order, with their resulting metrics.",
    )


class DirectSignature(dspy.Signature):
    """You improve DSPy module instructions directly from mistake history.

    The `history` input is ordered from oldest to newest. The final history entry is the current sample,
    and it already includes every edit attempt made on that sample so far.

    Produce targeted search/replace blocks for the current instructions.
    - Prefer small, concrete edits over full rewrites.
    - Reuse prior attempts in the history to avoid repeating ineffective edits.
    - Only emit edits for modules that should change.
    - If `search` is empty, `replace` will be appended.
    - If `replace` is empty, the matched `search` text will be deleted.
    """

    history: list[DirectHistoryEntry] = dspy.InputField(
        desc="Mistake history ordered from oldest to newest; the newest sample is last.",
    )
    current_instructions: dict[str, str] = dspy.InputField(
        desc="Current instructions for each module that may be edited.",
    )
    module_names: list[str] = dspy.InputField(
        desc="Ordered predictor names available for editing.",
    )
    module_edits: dict[str, list[SearchReplaceBlock]] = dspy.OutputField(
        desc="Per-module search/replace blocks to apply to the current instructions.",
    )


def _append_instruction_text(instructions: str, text: str) -> str:
    if not text:
        return instructions
    if not instructions.strip():
        return text
    return f"{instructions}\n\n{text}"


def _apply_instruction_edits(instructions: str, edits: list[SearchReplaceBlock]) -> tuple[str, bool]:
    updated = instructions
    applied = False

    for edit in edits:
        search = edit.search
        replace = edit.replace

        if not search:
            if replace:
                updated = _append_instruction_text(updated, replace)
                applied = True
            continue

        if search not in updated:
            logger.info("Direct edit search miss; skipping block.")
            continue

        updated = updated.replace(search, replace, 1)
        applied = True

    return updated, applied


def _apply_module_edits(program: Module, module_edits: dict[str, list[SearchReplaceBlock]]) -> bool:
    applied = False

    for name, predictor in program.named_predictors():
        edits = module_edits.get(name)
        if not edits:
            continue

        updated_instructions, did_apply = _apply_instruction_edits(predictor.signature.instructions, edits)
        if did_apply:
            predictor.signature = predictor.signature.with_instructions(updated_instructions)
            applied = True

    return applied


class Direct(Teleprompter):
    """Mistake-driven direct instruction optimizer.

    Direct walks the trainset sequentially. For each sample whose metric is <= 0, it asks an optimizer
    module for search/replace edits over the current predictor instructions. The optimizer keeps working on
    that sample for up to `max_iters_per_mistake` iterations before moving on to the next sample.
    """

    def __init__(
        self,
        *,
        metric,
        prompt_model=None,
        max_iters_per_mistake: int = 20,
        max_history_steps: int | None = None,
        max_history_tokens: int | None = None,
        trim_history_on_context_error: bool = True,
        use_cot: bool = False,
    ):
        super().__init__()
        if metric is None:
            raise ValueError("Direct requires a metric.")
        if max_iters_per_mistake < 1:
            raise ValueError("max_iters_per_mistake must be at least 1.")
        if max_history_steps is not None and max_history_steps < 1:
            raise ValueError("max_history_steps must be at least 1 when provided.")
        if max_history_tokens is not None and max_history_tokens < 1:
            raise ValueError("max_history_tokens must be at least 1 when provided.")

        self.metric = metric
        self.prompt_model = prompt_model
        self.max_iters_per_mistake = max_iters_per_mistake
        self.max_history_steps = max_history_steps
        self.max_history_tokens = max_history_tokens
        self.trim_history_on_context_error = trim_history_on_context_error
        self.use_cot = use_cot
        self.optimizer_module = (
            dspy.ChainOfThought(DirectSignature) if use_cot else dspy.Predict(DirectSignature)
        )

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
        **kwargs,
    ) -> Module:
        del teacher
        eval_kwargs = dict(kwargs.pop("eval_kwargs", {}) or {})
        eval_kwargs.pop("metric", None)
        eval_kwargs.pop("devset", None)
        del kwargs

        program = student.deepcopy()
        if not program.predictors():
            raise ValueError("Direct requires a student program with at least one predictor.")

        prompt_model = self._resolve_prompt_model(program)
        history: list[DirectHistoryEntry] = []
        stats = {
            "mistakes_seen": 0,
            "edits_applied": 0,
            "history_halvings": 0,
            "context_retries": 0,
        }

        for example in trainset:
            prediction, score = self._score_example(program, example)
            self._record_initial_train_metric(example, prediction, score)
            if score > 0:
                continue

            stats["mistakes_seen"] += 1
            history.append(
                DirectHistoryEntry(
                    sample=_summarize_sample_for_history(example.toDict()),
                    metric=score,
                )
            )

            for _ in range(self.max_iters_per_mistake):
                while self._history_exceeds_limits(history, prompt_model):
                    if not self._halve_history(history):
                        break
                    stats["history_halvings"] += 1

                module_edits = self._propose_edits(program, history, prompt_model, stats)
                if not module_edits:
                    break

                if not _apply_module_edits(program, module_edits):
                    break

                stats["edits_applied"] += 1
                updated_prediction, updated_score = self._score_example(program, example)
                self._record_reflection_metric(example, updated_prediction, updated_score)
                history[-1].edits.append(
                    DirectEditStep(
                        edits=module_edits,
                        resulting_metric=updated_score,
                    )
                )

                if updated_score > 0:
                    break

        program.direct_history = [entry.model_dump(mode="json") for entry in history]
        program.direct_stats = stats
        program.direct_valset_result = None
        program.direct_valset_score = None

        if valset:
            logger.info("Direct evaluating compiled program on valset (%d examples).", len(valset))
            evaluator = dspy.Evaluate(
                devset=valset,
                metric=lambda example, prediction: self._coerce_metric(self.metric(example, prediction)),
                **eval_kwargs,
            )
            program.direct_valset_result = evaluator(program)
            program.direct_valset_score = program.direct_valset_result.score

        return program

    def _record_initial_train_metric(
        self,
        example: Example,
        prediction: Prediction | None,
        score: float,
    ) -> None:
        recorder = getattr(self.metric, "record_initial_train_metric", None)
        if not callable(recorder):
            return

        try:
            recorder(example, prediction, score)
        except Exception as exc:
            logger.warning("Direct initial train metric recorder failed: %s", exc)

    def _record_reflection_metric(
        self,
        example: Example,
        prediction: Prediction | None,
        score: float,
    ) -> None:
        recorder = getattr(self.metric, "record_reflection_metric", None)
        if not callable(recorder):
            return

        try:
            recorder(example, prediction, score)
        except Exception as exc:
            logger.warning("Direct reflection metric recorder failed: %s", exc)

    def _resolve_prompt_model(self, student: Module):
        if self.prompt_model is not None:
            return self.prompt_model

        try:
            prompt_model = student.get_lm()
        except Exception:
            prompt_model = dspy.settings.lm

        if prompt_model is None:
            raise ValueError(
                "Direct could not resolve a prompt model. Pass `prompt_model=` or configure a global LM."
            )

        return prompt_model

    def _score_example(self, program: Module, example: Example) -> tuple[Prediction | None, float]:
        prediction = None
        try:
            prediction = program(**example.inputs())
        except Exception as exc:
            logger.warning("Direct failed to execute the student program on a training sample: %s", exc)

        try:
            metric_value = self.metric(example, prediction)
        except Exception as exc:
            logger.warning("Direct metric evaluation failed on a training sample: %s", exc)
            return prediction, 0.0

        return prediction, self._coerce_metric(metric_value)

    def _coerce_metric(self, metric_value: Any) -> float:
        if isinstance(metric_value, Prediction):
            if not hasattr(metric_value, "score"):
                raise ValueError("When Direct receives a Prediction from `metric`, it must contain a `score` field.")
            metric_value = metric_value.score

        if isinstance(metric_value, numbers.Real):
            return float(metric_value)

        raise TypeError(
            "Direct requires a numeric metric or a Prediction with a numeric `score` field. "
            f"Received {type(metric_value)}."
        )

    def _propose_edits(
        self,
        program: Module,
        history: list[DirectHistoryEntry],
        prompt_model,
        stats: dict[str, int],
    ) -> dict[str, list[SearchReplaceBlock]]:
        current_instructions = {
            name: predictor.signature.instructions
            for name, predictor in program.named_predictors()
        }
        module_names = list(current_instructions.keys())

        while True:
            try:
                with dspy.context(lm=prompt_model, trace=[]):
                    prediction = self.optimizer_module(
                        history=history,
                        current_instructions=current_instructions,
                        module_names=module_names,
                    )
                return prediction.module_edits
            except Exception as exc:
                if not self._should_retry_after_context_error(exc):
                    raise
                if not self._halve_history(history):
                    raise
                stats["context_retries"] += 1
                stats["history_halvings"] += 1

    def _should_retry_after_context_error(self, exc: Exception) -> bool:
        return self.trim_history_on_context_error and _is_context_window_error(exc)

    def _history_exceeds_limits(self, history: list[DirectHistoryEntry], prompt_model) -> bool:
        if self.max_history_steps is not None and self._history_step_count(history) > self.max_history_steps:
            return True

        if self.max_history_tokens is not None and self._history_token_count(history, prompt_model) > self.max_history_tokens:
            return True

        return False

    def _history_step_count(self, history: list[DirectHistoryEntry]) -> int:
        return sum(1 + len(entry.edits) for entry in history)

    def _history_token_count(self, history: list[DirectHistoryEntry], prompt_model) -> int:
        serialized_history = json.dumps(
            [entry.model_dump(mode="json") for entry in history],
            ensure_ascii=False,
        )
        model = getattr(prompt_model, "model", "")
        try:
            return litellm.token_counter(model=model, text=serialized_history)
        except Exception:
            return max(1, len(serialized_history) // 4)

    def _halve_history(self, history: list[DirectHistoryEntry]) -> bool:
        if len(history) > 1:
            del history[: len(history) // 2]
            return True

        if not history:
            return False

        entry = history[0]
        if len(entry.edits) <= 1:
            return False

        split_idx = len(entry.edits) // 2
        prior_metric = entry.edits[split_idx - 1].resulting_metric
        history[0] = entry.model_copy(
            update={
                "metric": prior_metric,
                "edits": entry.edits[split_idx:],
            }
        )
        return True


def _is_context_window_error(exc: Exception) -> bool:
    if isinstance(exc, litellm.ContextWindowExceededError):
        return True

    message = str(exc).lower()
    return "context window exceeded" in message or "maximum context length" in message or "context_length_exceeded" in message


def _summarize_sample_for_history(sample: Any) -> Any:
    if isinstance(sample, dict):
        return {str(key): _summarize_sample_for_history(value) for key, value in sample.items()}

    if isinstance(sample, list):
        return [_summarize_sample_for_history(value) for value in sample]

    if isinstance(sample, tuple):
        return [_summarize_sample_for_history(value) for value in sample]

    if isinstance(sample, (str, int, float, bool)) or sample is None:
        return sample

    if isinstance(sample, bytes):
        return f"<bytes:{len(sample)}>"

    if sample.__class__.__module__.startswith("dspy.adapters.types"):
        return repr(sample)

    if isinstance(sample, pydantic.BaseModel):
        return _summarize_sample_for_history(sample.model_dump(mode="json"))

    if hasattr(sample, "toDict") and callable(sample.toDict):
        return _summarize_sample_for_history(sample.toDict())

    return repr(sample)

from __future__ import annotations

import inspect
import json
import logging
import numbers
import time
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
        description="Search/replace blocks that were attempted on predictor instructions.",
    )
    resulting_metric: float = pydantic.Field(
        description="Metric value after applying the edits.",
    )
    accepted: bool = pydantic.Field(
        default=True,
        description="Whether the edits improved the metric and were kept in the live instructions.",
    )


class DirectHistoryEntry(pydantic.BaseModel):
    sample: dict[str, Any] = pydantic.Field(
        description="The sample currently being optimized.",
    )
    metric: float = pydantic.Field(
        description="Metric value for the sample before any edits in this history entry.",
    )
    edits: list[DirectEditStep] = pydantic.Field(
        default_factory=list,
        description="All edits attempted for this sample, in order, with their resulting metrics.",
    )


class DirectSignature(dspy.Signature):
    """You improve DSPy module instructions directly from optimization history.

    The `history` input is ordered from oldest to newest. The final history entry is the current sample,
    and it already includes every edit attempt made on that sample so far, including attempts that were
    reverted because they did not improve the metric.

    `current_instructions` contains only the currently accepted instructions. Failed edit attempts may
    appear in `history` even when they are not present in `current_instructions`.

    Produce targeted search/replace blocks for the current instructions.
    - Prefer small, concrete edits over full rewrites.
    - Reuse prior attempts in the history to avoid repeating ineffective edits.
    - Treat reverted history entries as failed experiments, not as active instructions.
    - Each history edit attempt includes whether it was accepted and kept.
    - Only emit edits for modules that should change.
    - If `search` is empty, `replace` will be appended.
    - If `replace` is empty, the matched `search` text will be deleted.
    """

    history: list[DirectHistoryEntry] = dspy.InputField(
        desc="Optimization history ordered from oldest to newest; failed edit attempts may still appear even if they were reverted.",
    )
    current_instructions: dict[str, str] = dspy.InputField(
        desc="Current accepted instructions for each module that may be edited.",
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


def _current_instructions(program: Module) -> dict[str, str]:
    return {
        name: predictor.signature.instructions
        for name, predictor in program.named_predictors()
    }


def _instruction_lengths(program: Module) -> tuple[dict[str, int], int]:
    lengths = {
        name: len(predictor.signature.instructions)
        for name, predictor in program.named_predictors()
    }
    return lengths, sum(lengths.values())


def _token_totals(*models) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    seen_model_ids = set()

    for model in models:
        if model is None or id(model) in seen_model_ids or not hasattr(model, "history"):
            continue
        seen_model_ids.add(id(model))
        for interaction in model.history:
            usage = interaction.get("usage", {}) or {}
            input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    return input_tokens, output_tokens


def _restore_instructions(program: Module, instructions: dict[str, str]) -> None:
    for name, predictor in program.named_predictors():
        if name not in instructions:
            continue
        predictor.signature = predictor.signature.with_instructions(instructions[name])


class Direct(Teleprompter):
    """Threshold-driven direct instruction optimizer.

    Direct walks the trainset sequentially. It skips samples whose metric is already at or above
    `min_metric_to_skip` unless that threshold is `None`. For each remaining sample, it asks an optimizer
    module for search/replace edits over the current predictor instructions. Strictly improving edits are
    kept; non-improving edits are reverted but still recorded in history. The optimizer keeps working on
    a sample for up to `max_iters_per_mistake` iterations before moving on to the next sample.
    """

    def __init__(
        self,
        *,
        metric,
        prompt_model=None,
        max_iters_per_mistake: int = 3,
        min_metric_to_skip: float | None = 1.0,
        max_history_steps: int | None = None,
        max_history_tokens: int | None = None,
        trim_history_on_context_error: bool = True,
        transient_error_retries: int = 8,
        transient_retry_backoff_s: float = 5.0,
        use_cot: bool = True,
    ):
        super().__init__()
        if metric is None:
            raise ValueError("Direct requires a metric.")
        if max_iters_per_mistake < 1:
            raise ValueError("max_iters_per_mistake must be at least 1.")
        if min_metric_to_skip is not None and (
            isinstance(min_metric_to_skip, bool) or not isinstance(min_metric_to_skip, numbers.Real)
        ):
            raise ValueError("min_metric_to_skip must be numeric or None.")
        if max_history_steps is not None and max_history_steps < 1:
            raise ValueError("max_history_steps must be at least 1 when provided.")
        if max_history_tokens is not None and max_history_tokens < 1:
            raise ValueError("max_history_tokens must be at least 1 when provided.")
        if transient_error_retries < 0:
            raise ValueError("transient_error_retries must be at least 0.")
        if transient_retry_backoff_s < 0:
            raise ValueError("transient_retry_backoff_s must be at least 0.")

        self.metric = metric
        self.prompt_model = prompt_model
        self.max_iters_per_mistake = max_iters_per_mistake
        self.min_metric_to_skip = None if min_metric_to_skip is None else float(min_metric_to_skip)
        self.max_history_steps = max_history_steps
        self.max_history_tokens = max_history_tokens
        self.trim_history_on_context_error = trim_history_on_context_error
        self.transient_error_retries = transient_error_retries
        self.transient_retry_backoff_s = float(transient_retry_backoff_s)
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
            "samples_optimized": 0,
            "samples_skipped": 0,
            "edits_applied": 0,
            "history_halvings": 0,
            "context_retries": 0,
            "transient_retries": 0,
        }

        for example in trainset:
            prediction, score = self._score_example(program, example)
            should_optimize = self._should_optimize_score(score)
            if should_optimize:
                stats["samples_optimized"] += 1
                history.append(
                    DirectHistoryEntry(
                        sample=_summarize_sample_for_history(example.toDict()),
                        metric=score,
                    )
                )
            else:
                stats["samples_skipped"] += 1
            self._record_initial_train_metric(example, prediction, score, program, history, prompt_model)
            if not should_optimize:
                continue

            current_score = score

            for _ in range(self.max_iters_per_mistake):
                while self._history_exceeds_limits(history, prompt_model):
                    if not self._halve_history(history):
                        break
                    stats["history_halvings"] += 1

                module_edits = self._propose_edits(program, history, prompt_model, stats)
                if not module_edits:
                    break

                prior_instructions = _current_instructions(program)
                if not _apply_module_edits(program, module_edits):
                    break

                updated_prediction, updated_score = self._score_example(program, example)
                accepted = updated_score > current_score
                history[-1].edits.append(
                    DirectEditStep(
                        edits=module_edits,
                        resulting_metric=updated_score,
                        accepted=accepted,
                    )
                )

                if not accepted:
                    _restore_instructions(program, prior_instructions)

                self._record_reflection_metric(example, updated_prediction, updated_score, program, history, prompt_model)

                if accepted:
                    stats["edits_applied"] += 1
                    current_score = updated_score
                    if not self._should_optimize_score(current_score):
                        break
                    continue

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

    def _should_optimize_score(self, score: float) -> bool:
        return self.min_metric_to_skip is None or score < self.min_metric_to_skip

    def _record_initial_train_metric(
        self,
        example: Example,
        prediction: Prediction | None,
        score: float,
        program: Module,
        history: list[DirectHistoryEntry],
        prompt_model,
    ) -> None:
        recorder = getattr(self.metric, "record_initial_train_metric", None)
        if not callable(recorder):
            return

        try:
            self._call_metric_recorder(recorder, example, prediction, score, program, history, prompt_model)
        except Exception as exc:
            logger.warning("Direct initial train metric recorder failed: %s", exc)

    def _record_reflection_metric(
        self,
        example: Example,
        prediction: Prediction | None,
        score: float,
        program: Module,
        history: list[DirectHistoryEntry],
        prompt_model,
    ) -> None:
        recorder = getattr(self.metric, "record_reflection_metric", None)
        if not callable(recorder):
            return

        try:
            self._call_metric_recorder(recorder, example, prediction, score, program, history, prompt_model)
        except Exception as exc:
            logger.warning("Direct reflection metric recorder failed: %s", exc)

    def _call_metric_recorder(
        self,
        recorder,
        example: Example,
        prediction: Prediction | None,
        score: float,
        program: Module,
        history: list[DirectHistoryEntry],
        prompt_model,
    ) -> None:
        current_instructions = _current_instructions(program)
        instruction_lengths, total_instruction_length = _instruction_lengths(program)
        student_model = self._resolve_student_model(program)
        total_input_tokens, total_output_tokens = _token_totals(student_model, prompt_model)
        kwargs = {
            "program": program,
            "current_instructions": current_instructions,
            "instruction_lengths": instruction_lengths,
            "total_instruction_length": total_instruction_length,
            "history_step_count": self._history_step_count(history),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }

        try:
            signature = inspect.signature(recorder)
        except (TypeError, ValueError):
            recorder(example, prediction, score)
            return

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        supported_kwargs = {
            name: value
            for name, value in kwargs.items()
            if accepts_kwargs or name in signature.parameters
        }
        recorder(example, prediction, score, **supported_kwargs)

    def _resolve_student_model(self, program: Module):
        try:
            student_model = program.get_lm()
        except Exception:
            student_model = dspy.settings.lm

        if student_model is None:
            student_model = dspy.settings.lm

        return student_model

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
        transient_retries = 0

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
                    if self._should_retry_after_transient_error(exc, transient_retries):
                        transient_retries += 1
                        stats["transient_retries"] += 1
                        delay_s = self.transient_retry_backoff_s * transient_retries
                        logger.warning(
                            "Direct optimizer transient error (%s/%s): %s. Retrying in %.1fs.",
                            transient_retries,
                            self.transient_error_retries,
                            exc,
                            delay_s,
                        )
                        if delay_s > 0:
                            time.sleep(delay_s)
                        continue
                    raise
                if not self._halve_history(history):
                    raise
                stats["context_retries"] += 1
                stats["history_halvings"] += 1

    def _should_retry_after_context_error(self, exc: Exception) -> bool:
        return self.trim_history_on_context_error and _is_context_window_error(exc)

    def _should_retry_after_transient_error(self, exc: Exception, retries_so_far: int) -> bool:
        return retries_so_far < self.transient_error_retries and _is_transient_api_error(exc)

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
        prior_metric = entry.metric
        for edit in entry.edits[:split_idx]:
            if edit.accepted:
                prior_metric = edit.resulting_metric
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


def _is_transient_api_error(exc: Exception) -> bool:
    current: Exception | None = exc
    seen = set()
    transient_names = {
        "timeout",
        "readtimeout",
        "connecttimeout",
        "connecterror",
        "apitimeouterror",
        "apiconnectionerror",
        "internalservererror",
        "serviceunavailableerror",
        "ratelimiterror",
    }
    transient_markers = (
        "timed out",
        "timeout",
        "connection refused",
        "connection reset",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "rate limit",
        "too many requests",
        "server disconnected",
    )

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.__class__.__name__.lower() in transient_names:
            return True

        message = str(current).lower()
        if any(marker in message for marker in transient_markers):
            return True

        current = current.__cause__ or current.__context__

    return False


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

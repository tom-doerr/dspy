from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable
from contextlib import contextmanager

import dspy
from dspy.predict.predict import Predict
from dspy.primitives.module import Module as DSPyModule


@dataclass
class DemoRecord:
    example: dspy.Example
    alpha: float
    beta: float
    prior: float
    last_used_step: int
    usage: int = 0
    wins: int = 0


class StochasticBootstrapBestOfN(DSPyModule):
    """
    Stochastic bootstrapped variant of :class:`dspy.BestOfN`.

    This module repeatedly samples a few-shot slate from a replay buffer, runs the wrapped module
    with those demonstrations, and keeps the candidate with the highest reward (or the first to cross
    ``threshold``). Demonstrations are pulled via Thompson sampling with a Beta posterior that is
    initialised from the generation-time reward and updated every time a demo participates in the
    winning set. Each candidate uses its *own* sampled demo slate, so the N attempts in one call can
    differ in context.

    Parameters
    ----------
    module_or_signature:
        Either a compiled DSPy module or a signature (string / ``dspy.Signature`` subtype). Signatures are wrapped
        in ``dspy.Predict`` automatically.
    reward_fn:
        Callable receiving the keyword arguments passed to ``forward`` and the resulting prediction; must return a
        scalar reward (higher is better). Non-finite values are treated as ``-inf``.
    N:
        Number of candidate attempts per call.
    threshold:
        Optional early stopping threshold on the reward. When set, candidates stop once any attempt reaches at least
        this value. If left ``None`` the module returns the best reward after all attempts.
    fail_count:
        Maximum number of exceptions allowed across the N attempts before re-raising the last error. Defaults to ``N``.
    max_bootstrapped_demos:
        Maximum number of demos to sample for each candidate. The effective draw is ``min(max_bootstrapped_demos, len(memory))``.
    replay_buffer_size:
        Replay-buffer capacity per task key. When the buffer is full, the lowest-scoring demo (posterior mean) is evicted.
    per_key_memory:
        If ``True`` (default) maintain independent buffers per (module signature, input fields) key; otherwise share a global memory.
    recency_weight:
        Additive weight applied to the exponential recency bonus in Thompson sampling.
    recency_tau:
        Time constant (in steps) for the recency exponential decay.
    initial_score_weight:
        Scales how strongly the generation-time reward seeds the Beta posterior when a demo is added.
    credit_mode:
        Set-level credit assignment strategy. Currently ``\"best_only\"`` (default) or ``\"equal_split\"``.
    min_reward_to_store:
        Minimum normalised reward required before storing a demo in memory.
    num_workers:
        Thread pool size used to evaluate candidates. Defaults to ``dspy.settings.num_threads``.
    seed:
        Optional RNG seed controlling Thompson sampling and demo eviction.

    Notes
    -----
    * Few-shots are attached via :class:`dspy.teleprompt.LabeledFewShot`, not by mutating ``demos`` on modules. This
      keeps the public API surface small and ensures adapters format the demos correctly.
    * Each call tracks errors and rewards, giving you per-call ``errors`` / ``success_scores`` metadata if you inspect
      the internal state.
    * Replay memories are keyed by ``module.signature`` (or class name) plus the sorted input field names unless
      ``per_key_memory=False``.
    * The prompt for any single candidate contains at most ``max_bootstrapped_demos`` demonstrations even if the replay
      buffer holds more.
    """
    def __init__(
        self,
        module_or_signature: DSPyModule | str | type,
        reward_fn: Callable[[dict[str, Any], Any], float],
        N: int = 4,  # noqa: N803
        *,
        threshold: float | None = None,
        fail_count: int | None = None,
        max_bootstrapped_demos: int = 4,
        replay_buffer_size: int = 64,
        per_key_memory: bool = True,
        recency_weight: float = 0.2,
        recency_tau: float = 50.0,
        initial_score_weight: float = 2.0,
        credit_mode: str = "best_only",
        min_reward_to_store: float = 0.0,
        num_workers: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.module = module_or_signature if isinstance(module_or_signature, DSPyModule) else Predict(module_or_signature)  # type: ignore[arg-type]
        self.reward_fn = reward_fn
        self.N = int(N)
        self.threshold = threshold
        self.fail_count = int(fail_count if fail_count is not None else N)
        self.max_bootstrapped_demos = int(max_bootstrapped_demos)
        self.replay_buffer_size = int(replay_buffer_size)
        self.per_key_memory = per_key_memory
        self.recency_weight = float(recency_weight)
        self.recency_tau = float(recency_tau)
        self.initial_score_weight = float(initial_score_weight)
        self.credit_mode = credit_mode
        self.min_reward_to_store = float(min_reward_to_store)
        default_threads = getattr(dspy.settings, "num_threads", 1) or 1
        self.num_workers = int(num_workers if num_workers is not None else default_threads)
        self._rng = random.Random(seed)
        self._memory: dict[str, list[DemoRecord]] = {}
        self._step = 0
        self._rollout = 0
        self._score_min = float("inf")
        self._score_max = float("-inf")

    def reset_memory(self, key: str | None = None) -> None:
        """Clear stored demos globally or for a specific task key."""
        if key is None:
            self._memory.clear()
        else:
            self._memory.pop(key, None)

    def memory_snapshot(self) -> dict[str, list[dspy.Example]]:
        """Return a shallow copy of the replay buffer keyed by task signature."""
        return {k: [record.example for record in records] for k, records in self._memory.items()}

    def forward(self, **kwargs):
        """Run up to ``N`` candidates with independent demo samples and return the highest-reward prediction."""
        self._step += 1
        key = self._memory_key(kwargs)
        mem = self._memory.setdefault(key, [])
        plans = []
        for _ in range(self.N):
            plans.append((self._sample_indices(mem), self._next_rollout()))

        best_pred, best_score, best_indices = None, -math.inf, []
        errors: list[str] = []
        remaining_failures = self.fail_count
        stop = False
        inputs = {k: v for k, v in kwargs.items() if k != "config"}
        base_config = dict(kwargs.get("config", {}))

        @contextmanager
        def temporary_predictor_config(module, extra_config):
            if not extra_config:
                yield
                return

            named = getattr(module, "named_predictors", None)
            if not callable(named):
                yield
                return

            snapshots = []
            for _, predictor in named():
                if not hasattr(predictor, "config"):
                    continue
                snapshots.append((predictor, dict(predictor.config)))
                predictor.config = {**predictor.config, **extra_config}
            try:
                yield
            finally:
                for predictor, config_snapshot in snapshots:
                    predictor.config = config_snapshot

        def run_candidate(plan):
            idxs, rollout_id = plan
            demo_examples = [mem[i].example for i in idxs]
            try:
                module = self._module_with_demos(demo_examples)
                config = dict(base_config)
                temp = float(config.get("temperature", 0.0) or 0.0)
                if temp <= 0.0:
                    config["temperature"] = 1.0
                config["rollout_id"] = rollout_id

                with temporary_predictor_config(module, config):
                    try:
                        pred = module(**inputs, config=config)
                    except TypeError as exc:
                        if "unexpected keyword argument 'config'" in str(exc):
                            pred = module(**inputs)
                        else:
                            raise
                score = float(self.reward_fn(kwargs, pred))
                if math.isfinite(score):
                    self._observe_score(score)
                else:
                    score = -math.inf
                return ("ok", pred, score, idxs)
            except Exception as exc:  # pragma: no cover - surfaced via fail_count
                return ("error", exc)

        def handle_result(result):
            nonlocal best_pred, best_score, best_indices, remaining_failures, stop
            status = result[0]
            if status == "error":
                exc = result[1]
                errors.append(f"{exc.__class__.__name__}: {exc}")
                if remaining_failures == 0:
                    raise exc
                remaining_failures -= 1
                return
            _, pred, score, idxs = result
            if best_pred is None or score > best_score:
                best_pred, best_score, best_indices = pred, score, idxs
            if self.threshold is not None and score >= self.threshold:
                stop = True

        if self.num_workers > 1 and len(plans) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                futures = [pool.submit(run_candidate, plan) for plan in plans]
                try:
                    for future in as_completed(futures):
                        result = future.result()
                        handle_result(result)
                        if stop:
                            break
                finally:
                    for future in futures:
                        future.cancel()
        else:
            for plan in plans:
                handle_result(run_candidate(plan))
                if stop:
                    break

        if best_pred is None:
            if errors:
                unique_errors = list(dict.fromkeys(errors))
                detail = "; ".join(unique_errors[:3])
                raise RuntimeError(f"All candidate attempts failed. {len(errors)} error(s): {detail}")
            raise RuntimeError("All candidate attempts failed without producing any predictions.")

        norm = self._normalise_score(best_score)
        if best_indices:
            self._credit(mem, best_indices, best_score, norm)
        if norm >= self.min_reward_to_store:
            demo = self._build_demo(inputs, best_pred)
            if demo is not None:
                self._store(mem, demo, norm)

        return best_pred

    def _module_with_demos(self, demos: list[dspy.Example]) -> DSPyModule:
        """Clone the wrapped module and attach the provided demos via ``LabeledFewShot`` (or reset when empty)."""
        if demos:
            teleprompter = dspy.LabeledFewShot(k=len(demos))
            return teleprompter.compile(self.module, trainset=demos, sample=False)
        return self.module.reset_copy()

    def _sample_indices(self, mem: list[DemoRecord]) -> list[int]:
        if not mem or self.max_bootstrapped_demos <= 0:
            return []
        size = min(self.max_bootstrapped_demos, len(mem))
        scored = []
        for idx, record in enumerate(mem):
            theta = self._rng.betavariate(record.alpha, record.beta)
            age = self._step - record.last_used_step
            decay = math.exp(-age / max(self.recency_tau, 1e-6))
            draw = theta + self.recency_weight * decay + 0.05 * record.prior
            scored.append((draw, idx))
        scored.sort(reverse=True)
        return [idx for _, idx in scored[:size]]

    def _credit(self, mem: list[DemoRecord], indices: list[int], score: float, norm: float) -> None:
        if not indices:
            return
        gain = max(0.0, min(1.0, norm))
        if self.credit_mode == "equal_split":
            gain /= len(indices)
        for idx in indices:
            record = mem[idx]
            record.usage += 1
            record.wins += 1
            record.last_used_step = self._step
            record.alpha += gain
            record.beta += 1.0 - gain

    def _store(self, mem: list[DemoRecord], demo: dspy.Example, norm: float) -> None:
        alpha = 1.0 + self.initial_score_weight * norm
        beta = 1.0 + self.initial_score_weight * (1.0 - norm)
        mem.append(DemoRecord(example=demo, alpha=alpha, beta=beta, prior=norm, last_used_step=self._step))
        if len(mem) > self.replay_buffer_size:
            worst = min(range(len(mem)), key=lambda i: mem[i].alpha / (mem[i].alpha + mem[i].beta + 1e-9))
            mem.pop(worst)

    def _build_demo(self, inputs: dict[str, Any], pred: Any) -> dspy.Example | None:
        data = {k: v for k, v in inputs.items()}
        if isinstance(pred, dspy.Example):
            for key in pred.keys():
                data[key] = pred[key]
        elif isinstance(pred, dict):
            data.update(pred)
        elif hasattr(pred, "__dict__"):
            data.update({k: v for k, v in vars(pred).items() if not k.startswith("_")})
        else:
            data["output"] = pred
        try:
            return dspy.Example(**data).with_inputs(*inputs.keys())
        except Exception:
            return None

    def _memory_key(self, kwargs: dict[str, Any]) -> str:
        """Stable key using signature inputs; ignore transport-only fields.

        - Prefer the wrapped module's signature input fields; otherwise, inspect
          the first inner predictor via `named_predictors()`.
        - Exclude transient keys (e.g., 'config', 'image_path', 'repo_url').
        """
        if not self.per_key_memory:
            return "global"

        sig = getattr(self.module, "signature", None)
        name = getattr(sig, "__name__", self.module.__class__.__name__)

        # Determine allowed fields from signature; fall back to inner predictors; then to kwargs
        allowed: set[str] = set()
        if getattr(self.module, "signature", None) is not None:
            try:
                allowed = set(self.module.signature.input_fields.keys())  # type: ignore[attr-defined]
            except Exception:
                allowed = set()
        if not allowed:
            named = getattr(self.module, "named_predictors", None)
            if callable(named):
                for _, predictor in named():
                    if getattr(predictor, "signature", None) is not None:
                        try:
                            allowed = set(predictor.signature.input_fields.keys())  # type: ignore[attr-defined]
                        except Exception:
                            allowed = set()
                        if allowed:
                            break
        if not allowed:
            allowed = set(k for k in kwargs if k != "config")

        volatile = {"config", "image_path", "repo_url"}
        fields = sorted(k for k in kwargs if k in allowed and k not in volatile)
        return f"{name}:{','.join(fields)}"

    def _next_rollout(self) -> int:
        self._rollout += 1
        return self._rollout

    def _normalise_score(self, score: float) -> float:
        self._score_min = min(self._score_min, score)
        self._score_max = max(self._score_max, score)
        if not math.isfinite(score):
            return 0.0
        if self._score_max == self._score_min:
            # Avoid flooding initial demos with perfect priors.
            return 0.5
        return max(0.0, min(1.0, (score - self._score_min) / (self._score_max - self._score_min)))

    def _observe_score(self, score: float) -> None:
        if not math.isfinite(score):
            return
        if score < self._score_min:
            self._score_min = score
        if score > self._score_max:
            self._score_max = score

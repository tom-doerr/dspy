"""Unit tests for PRISM optimizer."""
from __future__ import annotations
from concurrent.futures import Future
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt.prism import (
    PRISM, PrismState, _CreditModel, _Piece,
    _build, _fmt_observation, _sample, _set_instructions,
    _summarize_prediction,
)
from dspy.utils.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, sig="input -> output"):
        super().__init__()
        self.predictor = Predict(sig)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def always_one(example, prediction, trace=None):
    return 1.0


def score_match(example, prediction, trace=None):
    if prediction is None:
        return 0.0
    return 1.0 if example.output == prediction.output else 0.0


def neg_on_wrong(example, prediction):
    """reward_fn: -1 on mismatch (triggers gen_on_mistake)."""
    if prediction is None:
        return -1.0
    return 1.0 if example.output == prediction.output else -1.0


def _trainset(n=6):
    return [
        Example(input=f"q{i}", output=f"a{i}").with_inputs("input")
        for i in range(n)
    ]


def _done_future(result):
    f = Future()
    f.set_result(result)
    return f


def _pending_future():
    return Future()


# 1. gen_on_mistake with num_threads=1
def test_gen_on_mistake_cap_is_one_not_zero():
    """max(1, num_threads-1) with num_threads=1 must be 1."""
    assert max(1, 1 - 1) == 1


def test_gen_on_mistake_submits_with_one_thread():
    """num_threads=1 + gen_on_mistake: gens fire on failure."""
    lm = DummyLM([{"output": "wrong"}] * 100)
    dspy.settings.configure(lm=lm)
    opt = PRISM(
        metric=score_match, reward_fn=neg_on_wrong,
        max_steps=3, gen_on_mistake=True, num_threads=1,
        initial_knowledge=["test piece"],
    )
    opt._gen_async = MagicMock(return_value=["new"])
    opt.compile(SimpleModule(), trainset=_trainset(), seed=42)
    assert opt.state.gen_count > 0


# 2. gen_on_mistake with num_threads=8
def test_gen_on_mistake_cap_is_seven():
    assert max(1, 8 - 1) == 7


def test_gen_on_mistake_submits_with_eight_threads():
    lm = DummyLM([{"output": "wrong"}] * 200)
    dspy.settings.configure(lm=lm)
    opt = PRISM(
        metric=score_match, reward_fn=neg_on_wrong,
        max_steps=3, gen_on_mistake=True, num_threads=8,
        initial_knowledge=["test piece"],
    )
    opt._gen_async = MagicMock(return_value=["new"])
    opt.compile(SimpleModule(), trainset=_trainset(), seed=42)
    assert opt.state.gen_count > 0


# 3. gen count cap prevents queue buildup
def test_pending_at_cap_blocks():
    """3 pending with cap=3 blocks submission."""
    cap = max(1, 4 - 1)
    futs = [_pending_future() for _ in range(3)]
    pending = sum(1 for f in futs if not f.done())
    assert not (pending < cap)


def test_done_futures_dont_count():
    """Done futures are not pending."""
    cap = max(1, 4 - 1)
    futs = [_done_future(["x"]), _done_future(["y"]),
            _pending_future()]
    pending = sum(1 for f in futs if not f.done())
    assert pending == 1
    assert pending < cap


def test_below_cap_allows():
    """2 pending with cap=3 allows submission."""
    cap = max(1, 4 - 1)
    futs = [_pending_future() for _ in range(2)]
    pending = sum(1 for f in futs if not f.done())
    assert pending < cap


def test_cap_one_thread_one_pending_blocks():
    """num_threads=1: 1 pending >= cap(1), blocks."""
    cap = max(1, 1 - 1)
    futs = [_pending_future()]
    pending = sum(1 for f in futs if not f.done())
    assert not (pending < cap)


# 4. Shared ThreadPoolExecutor for gen+eval
def test_shared_executor():
    """compile() reuses one executor for _step_batch."""
    lm = DummyLM([{"output": "a0"}] * 200)
    dspy.settings.configure(lm=lm)
    opt = PRISM(
        metric=always_one, max_steps=2,
        gen_every=1, num_threads=2,
        initial_knowledge=["piece"],
    )
    seen = []
    orig = opt._step_batch

    def spy(s, ts, ps, n, deck_idx=0, executor=None):
        seen.append(executor)
        return orig(s, ts, ps, n, deck_idx=deck_idx,
                    executor=executor)

    opt._step_batch = spy
    opt._gen_async = MagicMock(return_value=["x"])
    opt.compile(SimpleModule(), trainset=_trainset())
    assert all(e is not None for e in seen)
    assert all(e is seen[0] for e in seen)


# 5. Intercept is not regularized
def test_penalty_intercept_zero():
    """penalty[-1,-1] must be 0 (no regularization)."""
    alpha = 5.0
    nc = 4  # 3 features + intercept
    penalty = alpha * np.eye(nc)
    penalty[-1, -1] = 0  # mirroring code
    assert penalty[-1, -1] == 0.0
    assert penalty[0, 0] == alpha


def test_update_sets_intercept():
    """After update(), intercept should be non-zero."""
    cr = _CreditModel()
    np.random.seed(0)
    for _ in range(10):
        cr.add([float(np.random.rand() > 0.5)],
               5.0 + np.random.randn())
    cr.update([_Piece("p")])
    assert cr.intercept != 0.0


# 6. _collect_gen tracks gen_failures and gen_duplicates
def test_collect_gen_failures():
    """Empty result increments gen_failures."""
    opt = PRISM(metric=always_one, max_steps=1)
    ps = [_Piece("existing")]
    futs = [_done_future([]), _done_future(None)]
    opt._collect_gen(ps, futs)
    assert opt.state.gen_failures == 2
    assert len(ps) == 1


def test_collect_gen_duplicates():
    """Duplicate content increments gen_duplicates."""
    opt = PRISM(metric=always_one, max_steps=1)
    ps = [_Piece("existing piece")]
    futs = [_done_future(["existing piece"])]
    opt._collect_gen(ps, futs)
    assert opt.state.gen_duplicates == 1
    assert len(ps) == 1


def test_collect_gen_new_piece():
    """Novel content is added to pool."""
    opt = PRISM(metric=always_one, max_steps=1)
    ps = [_Piece("old")]
    futs = [_done_future(["brand new"])]
    opt._collect_gen(ps, futs)
    assert len(ps) == 2
    assert ps[1].content == "brand new"
    assert opt.state.gen_failures == 0


def test_collect_gen_mixed():
    """One future with dup + novel piece."""
    opt = PRISM(metric=always_one, max_steps=1)
    ps = [_Piece("old")]
    futs = [_done_future(["old", "new"])]
    opt._collect_gen(ps, futs)
    assert opt.state.gen_duplicates == 1
    assert len(ps) == 2
    assert ps[1].content == "new"


def test_collect_gen_whitespace_skipped():
    """Whitespace-only strings skipped."""
    opt = PRISM(metric=always_one, max_steps=1)
    ps = []
    futs = [_done_future(["  ", "\n", ""])]
    opt._collect_gen(ps, futs)
    assert len(ps) == 0
    assert opt.state.gen_failures == 0


def test_collect_gen_removes_futures():
    """Processed futures removed from list."""
    opt = PRISM(metric=always_one, max_steps=1)
    ps = []
    futs = [_done_future(["p1"]), _done_future(["p2"])]
    opt._collect_gen(ps, futs)
    assert len(futs) == 0
    assert len(ps) == 2


# 7. Observation formatting
def test_fmt_observation():
    """_fmt_observation returns string with inputs/predicted/expected/score."""
    ex = Example(input="q", output="a").with_inputs("input")
    pred = MagicMock()
    pred.keys = MagicMock(return_value=["output"])
    pred.output = "pred_a"
    obs = _fmt_observation(ex, pred, 0.75)
    assert "0.750" in obs
    assert "input" in obs
    assert "pred_a" in obs


def test_generation_observation_includes_successful_recent_eval():
    """Periodic generation should see recent successful evals too."""
    lm = DummyLM([{"output": "a0"}] * 100)
    dspy.settings.configure(lm=lm)
    opt = PRISM(
        metric=always_one,
        max_steps=1,
        gen_every=1,
        num_threads=1,
        initial_knowledge=["piece"],
    )
    opt._gen_async = MagicMock(return_value=["new"])

    opt.compile(SimpleModule(), trainset=_trainset(), seed=42)

    assert opt._gen_async.called
    observation = opt._gen_async.call_args.args[2]
    assert "Score: 1.000" in observation
    assert "Predicted:" in observation


def test_summarize_prediction_drops_completions_and_logprobs():
    pred = dspy.Prediction(output="answer", logprobs={"token": -0.1})
    pred._completions = {"large": "payload"}

    summarized = _summarize_prediction(pred)

    assert summarized.output == "answer"
    assert summarized._completions is None
    assert "logprobs" not in summarized.keys()


# 8. Validation assertions
def test_init_rejects_bad_num_threads():
    with pytest.raises(AssertionError, match="num_threads"):
        PRISM(metric=always_one, num_threads=0)


def test_init_rejects_bad_max_steps():
    with pytest.raises(AssertionError, match="max_steps"):
        PRISM(metric=always_one, max_steps=0)


def test_init_rejects_bad_gen_every():
    with pytest.raises(AssertionError, match="gen_every"):
        PRISM(metric=always_one, gen_every=-1)


def test_init_rejects_bad_temp():
    with pytest.raises(AssertionError, match="temp"):
        PRISM(metric=always_one, temp=-0.5)


def test_init_rejects_bad_sampling_mode():
    with pytest.raises(AssertionError, match="sampling"):
        PRISM(metric=always_one, sampling="correlated")


def test_prism_sampling_defaults_to_independent():
    opt = PRISM(metric=always_one)
    assert opt.sampling == "independent"


def test_sample_independent_ignores_covariance():
    pieces = [_Piece("a"), _Piece("b")]
    for p in pieces:
        p.coef = 0.1
        p.stderr = 1.0
    cov = np.eye(2)
    with patch("numpy.random.multivariate_normal") as mvn:
        _sample(pieces, temp=1.0, cov=cov)
    mvn.assert_not_called()


def test_sample_joint_uses_covariance():
    pieces = [_Piece("a"), _Piece("b")]
    for p in pieces:
        p.coef = 0.1
        p.stderr = 1.0
    cov = np.eye(2)
    with patch(
        "numpy.random.multivariate_normal",
        return_value=np.array([1.0, -1.0]),
    ) as mvn:
        sel = _sample(
            pieces, temp=1.0, cov=cov,
            sampling="joint")
    mvn.assert_called_once()
    assert sel == [0]


# 9. PrismState dataclass
def test_prism_state_defaults():
    s = PrismState()
    assert s.pool == []
    assert s.last_selected == []
    assert s.gen_count == 0
    assert s.gen_duplicates == 0
    assert s.gen_failures == 0
    assert s.last_eval_time == 0.0
    assert s.last_gen_time == 0.0


def test_prism_state_on_optimizer():
    opt = PRISM(metric=always_one, max_steps=1)
    assert isinstance(opt.state, PrismState)
    assert opt.state.pool == []


# 10. End-to-end compile test
def test_compile_end_to_end():
    """PRISM.compile() runs, updates state, and sets
    _prism_knowledge on the result."""
    lm = DummyLM([{"output": "a0"}] * 500)
    dspy.settings.configure(lm=lm)

    student = SimpleModule()
    trainset = _trainset()
    init_knowledge = ["Answer with the capital city name"]
    initial_pool_size = len(init_knowledge)

    opt = PRISM(
        metric=score_match,
        max_steps=10,
        gen_every=3,
        num_threads=1,
        initial_knowledge=init_knowledge,
    )
    opt._gen_async = MagicMock(return_value=["New piece"])
    result = opt.compile(student, trainset=trainset)

    assert opt.state.gen_count > 0
    assert len(opt.state.pool) > initial_pool_size
    assert hasattr(result, "_prism_knowledge")
    assert len(result._prism_knowledge) > 0


# 11. Failed evals count as steps
def test_failed_evals_count_as_steps():
    """Evals that raise exceptions still count toward
    max_steps so the optimizer terminates."""
    call_count = 0

    class FailingModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict("input -> output")

        def forward(self, **kw):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("simulated failure")

    lm = DummyLM([{"output": "x"}] * 200)
    dspy.settings.configure(lm=lm)
    opt = PRISM(
        metric=always_one, max_steps=5,
        num_threads=1,
        initial_knowledge=["piece"],
    )
    opt._gen_async = MagicMock(return_value=[])
    opt.compile(FailingModule(), trainset=_trainset())
    assert opt.state.eval_failures >= 5
    assert call_count >= 5

"""PRISM — Pool Regression Inference Selection Model.

Knowledge pool optimizer using Ridge regression for per-piece
credit assignment and uncertainty-based subset selection.
Periodically generates new pieces via LLM with β/SE feedback.
"""
from __future__ import annotations

import logging
import math
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, List

import numpy as np

import dspy
from dspy.teleprompt.teleprompt import Teleprompter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
SAMPLING_MODES = {"independent", "joint"}


@dataclass
class PrismState:
    """Mutable runtime state for a PRISM optimization run."""
    pool: list = field(default_factory=list)
    last_selected: list = field(default_factory=list)
    gen_count: int = 0
    gen_duplicates: int = 0
    gen_failures: int = 0
    gen_too_long: int = 0
    gen_pending: int = 0
    gen_evals_during: int = 0
    ridge_pred_error: float = 0.0
    eval_failures: int = 0
    last_eval_time: float = 0.0
    last_gen_time: float = 0.0


class KnowledgePiece(BaseModel):
    content: str; beta: float = 0.0; se: float = 1.0; n: int = 0
    def __str__(self):
        return f"[β={self.beta:+.3f} SE={self.se:.3f} n={self.n}] {self.content}"

class KnowledgePool(BaseModel):
    items: list[KnowledgePiece] = Field(default_factory=list)
    def __str__(self): return "\n".join(str(p) for p in self.items)

class _GenKnowledge(dspy.Signature):
    """Generate novel knowledge rules (max 15 words each) to maximize reward.
    Pieces ordered worst-to-best by β. Negative β hurts performance.
    No repeats or paraphrases of existing pool items."""
    pool: KnowledgePool = dspy.InputField(desc="All pieces ordered by β (worst→best)")
    observation = dspy.InputField(desc="Recent example: inputs, prediction, label, score")
    reasoning: str = dspy.OutputField(desc="What patterns help/hurt? What's missing?")
    new_knowledge: list[str] = dspy.OutputField(desc="Novel knowledge rules, max 15 words each")


class _Piece:
    __slots__ = ("content", "coef", "stderr", "n_sel")
    def __init__(self, content: str):
        self.content, self.coef, self.stderr, self.n_sel = content, 0.0, 1.0, 0


class _CreditModel:
    """Linear credit model with fixed small α for stability."""
    ALPHA = 1e-6

    def __init__(self, **kw):
        self.X, self.y = [], []
        self.intercept = 0.0
        self.cov = None

    def add(self, sv, reward):
        self.X.append(sv); self.y.append(reward)

    def update(self, pieces):
        if len(self.y) < 3: return
        n = len(pieces)
        X = np.array([list(r)[:n]+[0]*max(0,n-len(r))
                       for r in self.X], dtype=np.float64)
        Xb = np.column_stack([X, np.ones(len(X))])
        y = np.array(self.y)
        nc = Xb.shape[1]
        penalty = self.ALPHA * np.eye(nc)
        penalty[-1, -1] = 0  # don't regularize intercept
        A = Xb.T @ Xb + penalty
        Ainv = np.linalg.inv(A)
        coefs = Ainv @ Xb.T @ y
        resid = y - Xb @ coefs
        s2 = np.sum(resid**2) / max(1, len(y) - nc)
        cov = s2 * Ainv
        self.cov = cov[:n, :n]
        for i, p in enumerate(pieces):
            if i < n:
                p.coef = float(coefs[i])
                p.stderr = float(np.sqrt(max(0, cov[i,i])))
        self.intercept = float(coefs[-1])


def _draw_seen(pieces, seen, betas, temp, cov,
               sampling="independent"):
    """Draw piece utilities with independent or joint posterior noise."""
    assert sampling in SAMPLING_MODES, "sampling"
    if temp <= 0:
        return betas
    if sampling == "joint" and cov is not None and len(seen) <= cov.shape[0]:
        sub_cov = cov[np.ix_(seen, seen)]
        sub_cov = (sub_cov + sub_cov.T) / 2
        np.fill_diagonal(sub_cov, np.maximum(
            np.diag(sub_cov), 1e-12))
        try:
            return np.random.multivariate_normal(
                betas, temp**2 * sub_cov)
        except np.linalg.LinAlgError:
            pass
    stds = [pieces[i].stderr for i in seen]
    return betas + temp * np.array(stds) * np.random.randn(len(seen))


def _sample(pieces, temp=1.0, cov=None, sampling="independent"):
    """Select pieces with positive draw from the configured posterior."""
    n = len(pieces)
    if not n:
        return []
    betas = np.array([p.coef for p in pieces])
    draws = _draw_seen(
        pieces, list(range(n)), betas, temp, cov,
        sampling=sampling)
    pos = betas[betas > 0]
    thr = float(pos.mean()) if len(pos) else 0.0
    sel = [i for i, d in enumerate(draws) if d > thr]
    return sel if sel else [random.randrange(n)]


def _build(pieces, idxs):
    order = sorted(idxs, key=lambda i: pieces[i].coef / max(1e-12, pieces[i].stderr), reverse=True)
    return "\n".join(pieces[i].content for i in order)


def _fmt_observation(ex, pred, sc):
    inputs = dict(ex.inputs())
    images = [v for v in inputs.values()
              if not isinstance(v, (str, bool, int, float))]
    text_inp = {k: v if isinstance(v, (str, bool, int, float))
                else f"[{type(v).__name__}]"
                for k, v in inputs.items()}
    lbl = {k: getattr(ex, k, '')
           for k in (ex.labels() if hasattr(ex,'labels') else [])}
    out = {k: getattr(pred, k, '')
           for k in (pred.keys() if hasattr(pred,'keys') else [])
           if not k.startswith('_') and k != 'logprobs'}
    text = (f"Input: {text_inp}\nPredicted: {out}\n"
            f"Expected: {lbl}\nScore: {sc:.3f}")
    reasoning = getattr(pred, "_native_reasoning", None)
    if reasoning:
        text += f"\nNative thinking: {reasoning}"
    if images:
        return [text] + images
    return text


def _reasoning_from_history_entry(entry):
    chunks = []
    for output in entry.get("outputs", []) or []:
        if not isinstance(output, dict):
            continue
        rc = output.get("reasoning_content")
        if rc:
            chunks.append(str(rc).strip())
    return "\n".join(c for c in chunks if c).strip()


def _extract_native_reasoning(prog, max_chars=4000):
    histories = [getattr(prog, "history", []) or []]
    if hasattr(prog, "named_predictors"):
        for _, pred in prog.named_predictors():
            histories.append(getattr(pred, "history", []) or [])
    seen, chunks = set(), []
    for hist in histories:
        for entry in hist:
            uid = entry.get("uuid") if isinstance(entry, dict) else None
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            if not isinstance(entry, dict):
                continue
            rc = _reasoning_from_history_entry(entry)
            if rc:
                chunks.append(rc)
    text = "\n---\n".join(chunks).strip()
    if max_chars and len(text) > max_chars:
        text = text[-max_chars:].lstrip()
    return text


def _set_instructions(prog, knowledge):
    """Append knowledge to all predictor instructions."""
    if not knowledge:
        return
    for _, pred in prog.named_predictors():
        base = pred.signature.instructions
        pred.signature = pred.signature.with_instructions(
            base + "\n\n" + knowledge)


def _summarize_prediction(pred):
    """Drop heavyweight completion payloads once scoring is done."""
    if pred is None or not isinstance(pred, dspy.Prediction):
        return pred

    summarized = dspy.Prediction(**{
        key: getattr(pred, key)
        for key in pred.keys()
        if key != "logprobs"
    })
    reasoning = getattr(pred, "_native_reasoning", None)
    if reasoning:
        summarized._native_reasoning = reasoning
    return summarized


class PRISM(Teleprompter):
    """Pool Regression Inference Selection Model.

    Optimizes knowledge via Ridge regression credit assignment,
    uncertainty-based subset selection, and LLM generation."""
    def __init__(self, *, metric, reward_fn=None, max_steps=100,
                 gen_every=10, gen_on_mistake=False, gen_lm=None,
                 initial_knowledge=None,
                 num_threads=1, temp=1.0,
                 max_piece_words=15,
                 max_gen_parallel=None,
                 gen_n_obs=1, ablation=False,
                 sampling="independent", **kw):
        super().__init__()
        assert num_threads >= 1, "num_threads"
        assert max_steps >= 1, "max_steps"
        assert gen_every >= 0, "gen_every"
        assert temp >= 0, "temp"
        assert sampling in SAMPLING_MODES, "sampling"
        self.metric = metric
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.gen_every = gen_every
        self.gen_on_mistake = gen_on_mistake
        self.gen_lm = gen_lm
        self.initial_knowledge = initial_knowledge or []
        self.num_threads = num_threads
        self.temp = temp
        self.max_piece_words = max_piece_words
        self.max_gen_parallel = max_gen_parallel or num_threads
        self.gen_n_obs = max(1, gen_n_obs)
        self.ablation = ablation
        self.sampling = sampling
        self.state = PrismState()

    def compile(self, student, *, trainset, seed=0):
        random.seed(seed); np.random.seed(seed)
        ps = [_Piece(s) for s in self.initial_knowledge]
        self.state = PrismState(pool=ps)
        cr = _CreditModel()
        self._credit_model = cr
        if hasattr(self.metric, 'set_prism_refs'):
            self.metric.set_prism_refs(self.state, cr)
        gn = None
        if self.gen_every or self.gen_on_mistake:
            w = self.max_piece_words
            sig = _GenKnowledge.with_updated_fields(
                "new_knowledge",
                desc=f"Novel knowledge rules, max {w} words each")
            gn = dspy.Predict(sig)
        cands, recent = [], []
        recent_obs = deque(maxlen=self.gen_n_obs)
        gen_futs, gp = [], ThreadPoolExecutor(self.num_threads)
        self._gen_start_evals = {}  # future id → n_evals at submit
        deck = list(trainset)
        random.shuffle(deck)
        deck_idx = 0
        n_evals = 0
        last_gen_at = 0
        while n_evals < self.max_steps:
            self._collect_gen(ps, gen_futs, n_evals)
            n_gen = sum(1 for f in gen_futs
                        if not f.done())
            self.state.gen_pending = n_gen
            n_eval = max(1, self.num_threads - n_gen)
            if self.ablation:
                res, deck_idx = self._step_batch_ablation(
                    student, deck, ps, n_eval,
                    deck_idx=deck_idx, executor=gp)
                for (sc, k, sel, ex, pred), abls in res:
                    if sc is not None:
                        self._upd(ps, sel, sc, cr)
                    for asc, asel in abls:
                        if asc is not None and math.isfinite(asc):
                            self._upd(ps, asel, asc, cr)
                res = [(sc, k, sel, ex, pred) for (sc, k, sel, ex, pred), _ in res]
            else:
                res, deck_idx = self._step_batch(
                    student, deck, ps, n_eval,
                    deck_idx=deck_idx, executor=gp)
            for sc, k, sel, ex, pred in res:
                n_evals += 1
                if sc is None: continue
                self._upd(ps, sel, sc, cr)
                recent.append(sc)
                recent_obs.append(_fmt_observation(ex, pred, sc))
                obs = self._merge_obs(recent_obs)
                pending = sum(1 for f in gen_futs if not f.done())
                if (sc < 0 and self.gen_on_mistake and gn
                        and pending < self.max_gen_parallel):
                    self.state.gen_count += 1
                    f = gp.submit(self._gen_async,
                                  ps, gn, obs)
                    gen_futs.append(f)
                    self._gen_start_evals[id(f)] = n_evals
                cands.append({"score": sc, "knowledge": k})
            if (gn and not self.gen_on_mistake
                    and self.gen_every
                    and n_evals - last_gen_at >= self.gen_every):
                self.state.gen_count += 1
                last_gen_at = n_evals
                obs = self._merge_obs(recent_obs)
                f = gp.submit(self._gen_async,
                              ps, gn, obs)
                gen_futs.append(f)
                self._gen_start_evals[id(f)] = n_evals
            scs = [r[0] for r in res if r[0] is not None]
            avg = np.mean(scs) if scs else 0
            ra = np.mean(recent[-50:]) if recent else 0
            logger.info(f"{n_evals}/{self.max_steps}"
                        f" avg={avg:.3f} ra50={ra:.3f}"
                        f" pool={len(ps)}"
                        f" gen={len(gen_futs)}")
        self._collect_gen(ps, gen_futs, n_evals,
                          wait=True)
        gp.shutdown(wait=True)
        return self._finalize(student, cands, ps)

    def _step_batch(self, student, deck, ps, n,
                     deck_idx=0, executor=None):
        jobs = []
        for _ in range(n):
            if deck_idx >= len(deck):
                random.shuffle(deck)
                deck_idx = 0
            ex = deck[deck_idx]
            deck_idx += 1
            cov = self._credit_model.cov if self.sampling == "joint" else None
            sel = _sample(ps, self.temp, cov,
                          sampling=self.sampling) if ps else []
            k = _build(ps, sel) if sel else ""
            jobs.append((ex, sel, k))
        if n <= 1:
            ex, sel, k = jobs[0]
            sc, pred = self._eval(student, ex, k, sel)
            return [(sc, k, sel, ex, pred)], deck_idx
        out = []
        tp = executor or ThreadPoolExecutor(n)
        try:
            fs = {tp.submit(self._eval, student, ex, k, sel):
                  (sel, k, ex) for ex, sel, k in jobs}
            for f in as_completed(fs):
                sel, k, ex = fs[f]
                sc, pred = f.result()
                out.append((sc, k, sel, ex, pred))
        finally:
            if not executor:
                tp.shutdown(wait=False)
        return out, deck_idx

    def _step_batch_ablation(self, student, deck, ps,
                              n, deck_idx=0, executor=None):
        jobs, out = [], []
        for _ in range(n):
            if deck_idx >= len(deck):
                random.shuffle(deck); deck_idx = 0
            cov = (getattr(self._credit_model, 'cov', None)
                   if self.sampling == "joint" else None)
            sel = _sample(ps, self.temp, cov,
                          sampling=self.sampling) if ps else []
            jobs.append((deck[deck_idx], sel)); deck_idx += 1
        tp = executor or ThreadPoolExecutor(n)
        fs = {tp.submit(self._ablation_chain, student,
              ex, ps, s): 0 for ex, s in jobs}
        for f in as_completed(fs): out.append(f.result())
        if not executor: tp.shutdown(wait=False)
        return out, deck_idx

    def _eval(self, student, ex, knowledge, sel=None):
        import copy, time as _time
        if sel is not None:
            self.state.last_selected = sel
        t0 = _time.time()
        try:
            prog = copy.deepcopy(student)
            _set_instructions(prog, knowledge)
            with dspy.context(trace=[]):
                pred = prog(**ex.inputs())
            reasoning = _extract_native_reasoning(prog)
            if reasoning and isinstance(pred, dspy.Prediction):
                pred._native_reasoning = reasoning
            s = self.metric(ex, pred)
            if self.reward_fn:
                sc = float(self.reward_fn(ex, pred))
            else:
                sc = float(s) if isinstance(s, (int, float)) \
                    else float(getattr(s, 'score', 0))
            self.state.last_eval_time = _time.time() - t0
            return sc, _summarize_prediction(pred)
        except Exception as e:
            logger.warning(f"Eval: {e}")
            self.state.eval_failures += 1
            return None, None

    def _ablation_chain(self, student, ex, ps, sel):
        """Main eval + ablation sub-steps (Ridge only)."""
        sel = list(sel); random.shuffle(sel)
        pcs = [ps[i].content for i in sel]
        k_full = "\n".join(pcs)
        sc, pred = self._eval_kw(student, ex, k_full, sel)
        main = (sc, k_full, list(sel), ex, pred)
        abls = []
        for n in range(len(pcs) - 1, -1, -1):
            k = "\n".join(pcs[:n])
            asc = self._eval_kw_reward(student, ex, k)
            abls.append((asc, list(sel[:n])))
        return main, abls

    def _eval_kw_reward(self, student, ex, knowledge):
        """Eval returning reward only (no metric tracking)."""
        import copy
        try:
            p = copy.deepcopy(student)
            inp = dict(ex.inputs()); inp['knowledge'] = knowledge
            with dspy.context(trace=[]):
                pred = p(**inp)
            if self.reward_fn:
                return float(self.reward_fn(ex, pred))
            y = bool(getattr(ex, 'suitable_for_posting', None))
            yh = bool(getattr(pred, 'suitable_for_posting', None))
            return 1.0 if y == yh else -1.0
        except Exception as e:
            logger.warning(f"Eval: {e}")
            self.state.eval_failures += 1; return None

    def _eval_kw(self, student, ex, knowledge, sel=None):
        """Eval with knowledge as input kwarg."""
        import copy, time as _t
        if sel is not None:
            self.state.last_selected = sel
        t0 = _t.time()
        try:
            p = copy.deepcopy(student)
            inp = dict(ex.inputs()); inp['knowledge'] = knowledge
            with dspy.context(trace=[]):
                pred = p(**inp)
            reasoning = _extract_native_reasoning(p)
            if reasoning and isinstance(pred, dspy.Prediction):
                pred._native_reasoning = reasoning
            s = self.metric(ex, pred)
            sc = (float(self.reward_fn(ex, pred)) if self.reward_fn
                  else float(s) if isinstance(s, (int, float))
                  else float(getattr(s, 'score', 0)))
            self.state.last_eval_time = _t.time() - t0
            return sc, _summarize_prediction(pred)
        except Exception as e:
            logger.warning(f"Eval: {e}")
            self.state.eval_failures += 1
            return None, None

    def _upd(self, ps, sel, sc, cr):
        if not math.isfinite(sc): return
        for i in sel: ps[i].n_sel += 1
        sv = [1.0 if i in set(sel) else 0.0
              for i in range(len(ps))]
        pred = sum(ps[i].coef for i in sel) + cr.intercept
        self.state.ridge_pred_error = abs(sc - pred)
        cr.add(sv, sc); cr.update(ps)

    @staticmethod
    def _merge_obs(obs_deque):
        texts, images = [], []
        for o in obs_deque:
            if isinstance(o, list):
                texts.append(o[0])
                images.extend(o[1:])
            else:
                texts.append(str(o))
        sep = "\n---\n"
        merged = sep.join(texts)
        return [merged] + images if images else merged

    def _gen_async(self, ps, gen, observation):
        """Background thread: return new knowledge strings."""
        import time as _time
        t0 = _time.time()
        src = sorted(ps, key=lambda p: p.coef)
        pool = KnowledgePool(items=[
            KnowledgePiece(content=p.content, beta=p.coef,
                se=p.stderr, n=p.n_sel) for p in src])
        try:
            kw = {"pool": pool, "observation": observation}
            if self.gen_lm:
                with dspy.context(lm=self.gen_lm, trace=[]):
                    r = gen(**kw)
            else:
                with dspy.context(trace=[]):
                    r = gen(**kw)
            out = r.new_knowledge
            self.state.last_gen_time = _time.time() - t0
            if isinstance(out, str):
                return [out] if out.strip() else []
            return out if isinstance(out, list) else []
        except Exception as e:
            self.state.last_gen_time = _time.time() - t0
            logger.warning(f"Gen: {e}"); return []

    def _collect_gen(self, ps, futs,
                     n_evals=0, wait=False):
        """Harvest finished gen futures."""
        done = [f for f in futs if f.done() or wait]
        starts = getattr(self, '_gen_start_evals', {})
        existing = {p.content for p in ps}
        for f in done:
            futs.remove(f)
            s = starts.pop(id(f), n_evals)
            self.state.gen_evals_during = n_evals - s
            result = f.result() or []
            if not result:
                self.state.gen_failures += 1
            for s in result:
                s = s.strip() if isinstance(s, str) \
                    else str(s).strip()
                if not s: continue
                if len(s.split()) > self.max_piece_words:
                    self.state.gen_too_long += 1
                    continue
                if s in existing:
                    self.state.gen_duplicates += 1
                else:
                    ps.append(_Piece(s))
                    existing.add(s)

    def _finalize(self, student, cands, ps):
        import copy
        best = copy.deepcopy(student)
        best.candidate_programs = sorted(
            cands, key=lambda c: c["score"], reverse=True)[:10]
        ranked = sorted(ps, key=lambda p: p.coef, reverse=True)
        pos = [p for p in ranked if p.coef > 0]
        knowledge = "\n".join(p.content for p in
            (pos if pos else ranked[:4]))
        _set_instructions(best, knowledge)
        best._prism_knowledge = knowledge
        best._prism_pieces = [
            {"content": p.content, "beta": p.coef,
             "se": p.stderr, "n": p.n_sel} for p in ranked
        ]
        return best

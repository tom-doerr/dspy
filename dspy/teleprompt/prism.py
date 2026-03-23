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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List

import numpy as np

import dspy
from dspy.teleprompt.teleprompt import Teleprompter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KnowledgePiece(BaseModel):
    content: str; beta: float = 0.0; se: float = 1.0; n: int = 0
    def __str__(self):
        return f"[β={self.beta:+.3f} SE={self.se:.3f} n={self.n}] {self.content}"

class KnowledgePool(BaseModel):
    items: list[KnowledgePiece] = Field(default_factory=list)
    def __str__(self): return "\n".join(str(p) for p in self.items)

class Rollout(BaseModel):
    """A single evaluation rollout with inputs, labels, and outputs."""
    model_config = {"arbitrary_types_allowed": True}
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    expected: dict = Field(default_factory=dict)
    score: float = 0.0

class _GenKnowledge(dspy.Signature):
    """Generate one novel knowledge piece to maximize reward (higher β = better).
    No repeats of existing pool items."""
    pool: KnowledgePool = dspy.InputField(desc="Pieces with β/SE/n")
    rollout: Rollout = dspy.InputField(desc="Recent example with score")
    reasoning: str = dspy.OutputField(desc="What patterns help/hurt? What's missing?")
    new_knowledge: str = dspy.OutputField(desc="One novel knowledge string")


class _Piece:
    __slots__ = ("content", "coef", "stderr", "n_sel")
    def __init__(self, content: str):
        self.content, self.coef, self.stderr, self.n_sel = content, 0.0, 1.0, 0


class _CreditModel:
    """Ridge with CV-optimized α: reward ~ Σ piece_i."""
    def __init__(self, **kw):
        self.X, self.y = [], []
        self.log_alpha = 0.0
        self.intercept = 0.0

    def add(self, sv, reward):
        self.X.append(sv); self.y.append(reward)

    def _loo_cv(self, Xb, y, alpha):
        penalty = alpha * np.eye(Xb.shape[1])
        penalty[-1, -1] = 0
        A = Xb.T @ Xb + penalty
        H = Xb @ np.linalg.inv(A) @ Xb.T
        resid = y - H @ y
        h = np.diag(H)
        return np.mean((resid / (1 - h + 1e-10)) ** 2)

    def update(self, pieces):
        if len(self.y) < 3: return
        n = len(pieces)
        X = np.array([list(r)[:n]+[0]*max(0,n-len(r))
                       for r in self.X], dtype=np.float64)
        Xb = np.column_stack([X, np.ones(len(X))])
        y = np.array(self.y)
        cv0 = self._loo_cv(Xb, y, np.exp(self.log_alpha))
        for _ in range(5):
            la = self.log_alpha + np.random.randn() * 0.5
            cv = self._loo_cv(Xb, y, np.exp(la))
            if cv < cv0:
                self.log_alpha = la; cv0 = cv
        alpha = np.exp(self.log_alpha)
        nc = Xb.shape[1]
        penalty = alpha * np.eye(nc)
        penalty[-1, -1] = 0  # don't regularize intercept
        A = Xb.T @ Xb + penalty
        Ainv = np.linalg.inv(A)
        coefs = Ainv @ Xb.T @ y
        resid = y - Xb @ coefs
        s2 = np.sum(resid**2) / max(1, len(y) - nc)
        cov = s2 * Ainv
        for i, p in enumerate(pieces):
            if i < n:
                p.coef = float(coefs[i])
                p.stderr = float(np.sqrt(max(0, cov[i,i])))
        self.intercept = float(coefs[-1])


def _sample(pieces, temp=1.0):
    """Select pieces with positive draw. Unseen always included."""
    sel = []
    for i, p in enumerate(pieces):
        if p.n_sel == 0: sel.append(i); continue
        draw = p.coef + temp * p.stderr * np.random.randn() if temp > 0 else p.coef
        if draw > 0: sel.append(i)
    return sel if sel else [random.randrange(len(pieces))]


def _build(pieces, idxs):
    return "\n".join(pieces[i].content for i in idxs)


def _fmt_rollout(ex, pred, sc):
    inp = dict(ex.inputs())
    lbl = {k: getattr(ex, k, '')
           for k in (ex.labels() if hasattr(ex,'labels') else [])}
    out = {k: getattr(pred, k, '')
           for k in (pred.keys() if hasattr(pred,'keys') else [])
           if not k.startswith('_')}
    return Rollout(score=sc, input=inp,
                   expected=lbl, output=out)


def _set_instructions(prog, knowledge):
    """Append knowledge to all predictor instructions."""
    if not knowledge:
        return
    for _, pred in prog.named_predictors():
        base = pred.signature.instructions
        pred.signature = pred.signature.with_instructions(
            base + "\n\n" + knowledge)


class PRISM(Teleprompter):
    """Pool Regression Inference Selection Model.

    Optimizes knowledge via Ridge regression credit assignment,
    uncertainty-based subset selection, and LLM generation."""
    def __init__(self, *, metric, reward_fn=None, max_steps=100,
                 gen_every=10, gen_on_fail=False, gen_lm=None,
                 initial_knowledge=None,
                 num_threads=1, temp=1.0, **kw):
        super().__init__()
        self.metric = metric
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.gen_every = gen_every
        self.gen_on_fail = gen_on_fail
        self.gen_lm = gen_lm
        self.initial_knowledge = initial_knowledge or []
        self.num_threads = num_threads
        self.temp = temp
        self.pool = []
        self.last_selected = []
        self.gen_count = 0
        self.gen_duplicates = 0
        self.gen_failures = 0

    def compile(self, student, *, trainset, seed=0):
        random.seed(seed); np.random.seed(seed)
        ps = [_Piece(s) for s in self.initial_knowledge]
        self.pool = ps
        cr = _CreditModel()
        self._credit_model = cr
        gn = dspy.Predict(_GenKnowledge) if self.gen_every else None
        cands, last_fail, recent = [], Rollout(), []
        gen_futs, gp = [], ThreadPoolExecutor(self.num_threads)
        for i in range(self.max_steps):
            self._collect_gen(ps, gen_futs)
            n_gen = sum(1 for f in gen_futs if not f.done())
            n_eval = max(1, self.num_threads - n_gen)
            res = self._step_batch(student, trainset, ps,
                                   n_eval, executor=gp)
            for sc, k, sel, ex, pred in res:
                if sc is None: continue
                self.last_selected = sel
                self._upd(ps, sel, sc, cr)
                recent.append(sc)
                if sc < 0:
                    last_fail = _fmt_rollout(ex, pred, sc)
                    if self.gen_on_fail and gn:
                        self.gen_count += 1
                        gen_futs.append(gp.submit(
                            self._gen_async, ps, gn, last_fail))
                cands.append({"score": sc, "knowledge": k})
            if gn and not self.gen_on_fail and (i+1)%self.gen_every==0:
                self.gen_count += 1
                gen_futs.append(gp.submit(
                    self._gen_async, ps, gn, last_fail))
            scs = [r[0] for r in res if r[0] is not None]
            avg = np.mean(scs) if scs else 0
            ra = np.mean(recent[-50:]) if recent else 0
            logger.info(f"{i+1}/{self.max_steps} avg={avg:.3f}"
                        f" ra50={ra:.3f} pool={len(ps)}"
                        f" gen={len(gen_futs)}")
        self._collect_gen(ps, gen_futs, wait=True)
        gp.shutdown(wait=True)
        return self._finalize(student, cands, ps)

    def _step_batch(self, student, trainset, ps, n, executor=None):
        jobs = []
        for _ in range(n):
            ex = random.choice(trainset)
            sel = _sample(ps, self.temp) if ps else []
            k = _build(ps, sel) if sel else ""
            jobs.append((ex, sel, k))
        if n <= 1:
            ex, sel, k = jobs[0]
            sc, pred = self._eval(student, ex, k)
            return [(sc, k, sel, ex, pred)]
        out = []
        tp = executor or ThreadPoolExecutor(n)
        try:
            fs = {tp.submit(self._eval, student, ex, k):
                  (sel, k, ex) for ex, sel, k in jobs}
            for f in as_completed(fs):
                sel, k, ex = fs[f]
                sc, pred = f.result()
                out.append((sc, k, sel, ex, pred))
        finally:
            if not executor:
                tp.shutdown(wait=False)
        return out

    def _eval(self, student, ex, knowledge):
        import copy
        try:
            prog = copy.deepcopy(student)
            _set_instructions(prog, knowledge)
            pred = prog(**ex.inputs())
            s = self.metric(ex, pred)
            if self.reward_fn:
                sc = float(self.reward_fn(ex, pred))
            else:
                sc = float(s) if isinstance(s, (int, float)) \
                    else float(getattr(s, 'score', 0))
            return sc, pred
        except Exception as e:
            logger.warning(f"Eval: {e}"); return None, None

    def _upd(self, ps, sel, sc, cr):
        if not math.isfinite(sc): return
        for i in sel: ps[i].n_sel += 1
        sv = [1.0 if i in set(sel) else 0.0 for i in range(len(ps))]
        cr.add(sv, sc); cr.update(ps)

    def _gen_async(self, ps, gen, rollout):
        """Background thread: return new knowledge strings."""
        src = [p for p in ps if p.coef > 0 or p.n_sel == 0]
        pool = KnowledgePool(items=[
            KnowledgePiece(content=p.content, beta=p.coef,
                se=p.stderr, n=p.n_sel) for p in src])
        try:
            kw = {"pool": pool, "rollout": rollout}
            if self.gen_lm:
                with dspy.context(lm=self.gen_lm):
                    r = gen(**kw)
            else:
                r = gen(**kw)
            out = r.new_knowledge
            if isinstance(out, str):
                return [out] if out.strip() else []
            return out if isinstance(out, list) else []
        except Exception as e:
            logger.warning(f"Gen: {e}"); return []

    def _collect_gen(self, ps, futs, wait=False):
        """Harvest finished gen futures into the pool."""
        done = [f for f in futs if f.done() or wait]
        existing = {p.content for p in ps}
        for f in done:
            futs.remove(f)
            result = f.result() or []
            if not result:
                self.gen_failures += 1
            for s in result:
                s = s.strip() if isinstance(s, str) \
                    else str(s).strip()
                if not s: continue
                if s in existing:
                    self.gen_duplicates += 1
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

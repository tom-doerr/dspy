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

class _GenKnowledge(dspy.Signature):
    """Generate novel knowledge to maximize reward (higher β = better).
    Generate SHORT, CONCISE, DIFFERENT pieces — no repeats."""
    pool: KnowledgePool = dspy.InputField(desc="Pieces with β/SE/n")
    rollouts: str = dspy.InputField(desc="Recent examples: inputs, expected output, model output, score")
    new_knowledge: list[str] = dspy.OutputField(desc="3 short novel strings")


class _Piece:
    __slots__ = ("content", "coef", "stderr", "n_sel")
    def __init__(self, content: str):
        self.content, self.coef, self.stderr, self.n_sel = content, 0.0, 1.0, 0


class _CreditModel:
    """Linear regression credit assignment: reward ~ Σ piece_i."""
    def __init__(self, alpha=1.0, reg="ridge"):
        self.X, self.y, self.alpha = [], [], alpha
        self.reg = reg  # "ridge" (L2) or "lasso" (L1)

    def add(self, sv, reward):
        self.X.append(sv); self.y.append(reward)

    def update(self, pieces):
        if len(self.y) < 3: return
        n = len(pieces)
        X = np.array([list(r)[:n]+[0]*max(0,n-len(r)) for r in self.X], dtype=np.float64)
        y = np.array(self.y)
        if self.reg == "lasso":
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=1000)
        else:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=self.alpha, fit_intercept=True)
        model.fit(X, y)
        resid = y - model.predict(X)
        s2 = np.sum(resid**2) / max(1, len(y) - n - 1)
        XtX = X.T @ X
        A_inv = np.linalg.inv(XtX + self.alpha * np.eye(n))
        cov = s2 * (A_inv @ XtX @ A_inv)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        for i, p in enumerate(pieces):
            if i < len(model.coef_):
                p.coef = float(model.coef_[i])
                p.stderr = float(se[i])


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
    skip = (dspy.Image, bytes, memoryview)
    inp = {k: str(v)[:500] for k, v in ex.inputs().items()
           if not isinstance(v, skip)}
    lbl = {k: str(getattr(ex, k, ''))[:500]
           for k in (ex.labels() if hasattr(ex,'labels') else [])
           if not isinstance(getattr(ex, k, None), skip)}
    out = {k: str(getattr(pred, k, ''))[:500]
           for k in (pred.keys() if hasattr(pred,'keys') else [])
           if not k.startswith('_')}
    return (f"Score: {sc:.3f}\n  Input: {inp}"
            f"\n  Expected: {lbl}\n  Got: {out}")


class PRISM(Teleprompter):
    """Pool Regression Inference Selection Model.

    Optimizes knowledge via Ridge regression credit assignment,
    uncertainty-based subset selection, and LLM generation."""
    def __init__(self, *, metric, reward_fn=None, max_steps=100,
                 gen_every=10, gen_lm=None, initial_knowledge=None,
                 num_threads=1, temp=1.0, reg="ridge", alpha=1.0):
        super().__init__()
        self.metric = metric
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.gen_every = gen_every
        self.gen_lm = gen_lm
        self.initial_knowledge = initial_knowledge or []
        self.num_threads = num_threads
        self.temp = temp
        self.reg = reg
        self.alpha = alpha

    def compile(self, student, *, trainset, seed=0):
        random.seed(seed); np.random.seed(seed)
        ps = [_Piece(s) for s in self.initial_knowledge]
        cr = _CreditModel(alpha=self.alpha, reg=self.reg)
        gn = dspy.Predict(_GenKnowledge) if self.gen_every else None
        cands, last_ro, recent = [], "", []
        gen_futs, gp = [], ThreadPoolExecutor(self.num_threads)
        for i in range(self.max_steps):
            self._collect_gen(ps, gen_futs)
            n_gen = sum(1 for f in gen_futs if not f.done())
            n_eval = max(1, self.num_threads - n_gen)
            res = self._step_batch(student, trainset, ps,
                                   n_eval)
            for sc, k, sel, ex, pred in res:
                if sc is None: continue
                self._upd(ps, sel, sc, cr)
                recent.append(sc)
                last_ro = _fmt_rollout(ex, pred, sc)
                cands.append({"score": sc, "knowledge": k})
            if gn and (i+1)%self.gen_every==0:
                gen_futs.append(gp.submit(
                    self._gen_async, ps, gn, last_ro))
            scs = [r[0] for r in res if r[0] is not None]
            avg = np.mean(scs) if scs else 0
            ra = np.mean(recent[-50:]) if recent else 0
            logger.info(f"{i+1}/{self.max_steps} avg={avg:.3f}"
                        f" ra50={ra:.3f} pool={len(ps)}"
                        f" gen={len(gen_futs)}")
        self._collect_gen(ps, gen_futs, wait=True)
        gp.shutdown(wait=True)
        return self._finalize(student, cands, ps)

    def _step_batch(self, student, trainset, ps, n):
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
        with ThreadPoolExecutor(n) as tp:
            fs = {tp.submit(self._eval, student, ex, k):
                  (sel, k, ex) for ex, sel, k in jobs}
            for f in as_completed(fs):
                sel, k, ex = fs[f]
                sc, pred = f.result()
                out.append((sc, k, sel, ex, pred))
        return out

    def _eval(self, student, ex, knowledge):
        try:
            pred = student(knowledge=knowledge, **ex.inputs())
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
        for i in sel: ps[i].n_sel += 1
        sv = [1.0 if i in set(sel) else 0.0 for i in range(len(ps))]
        cr.add(sv, sc); cr.update(ps)

    def _gen_async(self, ps, gen, rollout):
        """Background thread: return new knowledge strings."""
        pool = KnowledgePool(items=[
            KnowledgePiece(content=p.content, beta=p.coef,
                se=p.stderr, n=p.n_sel) for p in ps])
        try:
            kw = {"pool": pool, "rollouts": rollout}
            if self.gen_lm:
                with dspy.context(lm=self.gen_lm):
                    r = gen(**kw)
            else:
                r = gen(**kw)
            out = r.new_knowledge
            return out if isinstance(out, list) else []
        except Exception as e:
            logger.warning(f"Gen: {e}"); return []

    def _collect_gen(self, ps, futs, wait=False):
        """Harvest finished gen futures into the pool."""
        done = [f for f in futs if f.done() or wait]
        existing = {p.content for p in ps}
        for f in done:
            futs.remove(f)
            for s in (f.result() or []):
                s = s.strip() if isinstance(s, str) \
                    else str(s).strip()
                if s and s not in existing:
                    ps.append(_Piece(s))
                    existing.add(s)

    def _finalize(self, student, cands, ps):
        import copy
        best = copy.deepcopy(student)
        best.candidate_programs = sorted(
            cands, key=lambda c: c["score"], reverse=True)[:10]
        ranked = sorted(ps, key=lambda p: p.coef, reverse=True)
        pos = [p for p in ranked if p.coef > 0]
        best._prism_knowledge = "\n".join(p.content for p in
            (pos if pos else ranked[:4]))
        best._prism_pieces = [
            {"content": p.content, "beta": p.coef,
             "se": p.stderr, "n": p.n_sel} for p in ranked
        ]
        return best

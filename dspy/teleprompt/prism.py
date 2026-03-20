"""PRISM — Pool Regression Inference Selection Model.

Knowledge pool optimizer using Ridge regression for per-piece
credit assignment and uncertainty-based subset selection.
Periodically generates new pieces via LLM with β/SE feedback.
"""
from __future__ import annotations

import logging
import math
import random
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

class NewKnowledge(BaseModel):
    items: list[str] = Field(default_factory=list)


class _GenKnowledge(dspy.Signature):
    """Generate novel knowledge to maximize reward (higher β = better).
    Generate SHORT, CONCISE, DIFFERENT pieces — no repeats."""
    pool: KnowledgePool = dspy.InputField(desc="Pieces with β/SE/n")
    rollout: str = dspy.InputField(desc="Last rollout: input, knowledge used, output, score")
    new_knowledge: NewKnowledge = dspy.OutputField(desc="Short novel strings")


class _Piece:
    __slots__ = ("content", "coef", "stderr", "n_sel")
    def __init__(self, content: str):
        self.content, self.coef, self.stderr, self.n_sel = content, 0.0, 1.0, 0


class _CreditModel:
    """Ridge regression with intercept: reward ~ Σ piece_i."""
    def __init__(self, alpha=1.0):
        self.X, self.y, self.alpha = [], [], alpha

    def add(self, sv, reward):
        self.X.append(sv); self.y.append(reward)

    def update(self, pieces):
        if len(self.y) < 3: return
        n = len(pieces)
        Xr = np.array([list(r)[:n]+[0]*max(0,n-len(r)) for r in self.X], dtype=np.float64)
        X = np.hstack([np.ones((len(Xr),1)), Xr])
        y, m = np.array(self.y), n+1
        A = X.T@X + self.alpha*np.eye(m)
        A[0,0] = (X[:,0]**2).sum()
        try: c = np.linalg.solve(A, X.T@y)
        except np.linalg.LinAlgError: return
        s2 = np.sum((y-X@c)**2)/max(1,len(y)-m)
        try: se = np.sqrt(np.maximum(np.diag(s2*np.linalg.inv(A)),1e-8))
        except np.linalg.LinAlgError: se = np.ones(m)
        for i, p in enumerate(pieces):
            if i+1 < len(c): p.coef, p.stderr = float(c[i+1]), float(se[i+1])


def _sample(pieces, temp=1.0):
    """Select pieces with positive uncertainty draw. temp scales SE.
    Unseen pieces (n_sel=0) are always included for initial evaluation."""
    sel = [i for i, p in enumerate(pieces)
           if p.n_sel == 0 or p.coef + temp * p.stderr * np.random.randn() > 0]
    return sel if sel else [random.randrange(len(pieces))]


def _build(pieces, idxs):
    return "\n".join(pieces[i].content for i in idxs)


class PRISM(Teleprompter):
    """Pool Regression Inference Selection Model.

    Optimizes knowledge via Ridge regression credit assignment,
    uncertainty-based subset selection, and LLM generation."""
    def __init__(self, *, metric, max_steps=100, gen_every=10,
                 gen_lm=None, initial_knowledge=None, num_threads=1, temp=1.0):
        super().__init__()
        self.metric = metric
        self.max_steps = max_steps
        self.gen_every = gen_every
        self.gen_lm = gen_lm
        self.initial_knowledge = initial_knowledge or []
        self.num_threads = num_threads
        self.temp = temp

    def compile(self, student, *, trainset, seed=0):
        random.seed(seed); np.random.seed(seed)
        ps = [_Piece(s) for s in self.initial_knowledge]
        cr, gn = _CreditModel(), dspy.Predict(_GenKnowledge) if self.gen_every else None
        bs, bk, cands, last_ro = -math.inf, "", [], ""
        for i in range(self.max_steps):
            sc, k, ex, pred = self._step(student, trainset, ps, cr)
            if sc is None: continue
            last_ro = f"Score={sc:.4f}\nKnowledge: {k}\nInput: {ex.inputs()}\nOutput: {pred}"
            cands.append({"score": sc, "knowledge": k})
            if sc > bs: bs, bk = sc, k
            if gn and (i+1)%self.gen_every==0: self._gen(ps, gn, last_ro)
            if (i+1)%10==0: logger.info(f"{i+1}/{self.max_steps} best={bs:.4f} pool={len(ps)}")
        return self._fin(student, bk, cands)

    def _step(self, student, trainset, ps, cr):
        ex = random.choice(trainset)
        sel = _sample(ps, self.temp) if ps else []
        k = _build(ps, sel) if sel else ""
        sc, pred = self._eval(student, ex, k)
        if sc is not None: self._upd(ps, sel, sc, cr)
        return sc, k, ex, pred

    def _eval(self, student, ex, knowledge):
        try:
            pred = student(knowledge=knowledge, **ex.inputs())
            s = self.metric(ex, pred)
            sc = float(s) if isinstance(s, (int, float)) else float(getattr(s, 'score', 0))
            return sc, pred
        except Exception as e:
            logger.warning(f"Eval: {e}"); return None, None

    def _upd(self, ps, sel, sc, cr):
        for i in sel: ps[i].n_sel += 1
        sv = [1.0 if i in set(sel) else 0.0 for i in range(len(ps))]
        cr.add(sv, sc); cr.update(ps)

    def _gen(self, ps, gen, rollout):
        pool = KnowledgePool(items=[
            KnowledgePiece(content=p.content, beta=p.coef, se=p.stderr, n=p.n_sel) for p in ps])
        kw = {"pool": pool, "rollout": rollout}
        if self.gen_lm: kw["lm"] = self.gen_lm
        try:
            r = gen(**kw)
            new = r.new_knowledge.items if hasattr(r.new_knowledge, 'items') else []
            existing = {p.content for p in ps}
            for s in new:
                s = s.strip() if isinstance(s, str) else str(s).strip()
                if s and s not in existing: ps.append(_Piece(s)); existing.add(s)
        except Exception as e:
            logger.warning(f"Gen: {e}")

    def _fin(self, student, best_k, cands):
        import copy
        best = copy.deepcopy(student)
        best.candidate_programs = sorted(cands, key=lambda c: c["score"], reverse=True)[:10]
        best._prism_knowledge = best_k
        best._prism_pieces = None  # set below if available
        return best

"""Teacher-Forced Log-Likelihood (TFLL) metric for DSPy."""

from dataclasses import dataclass
import math
from typing import Callable, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

START, END = "<<<ANS>>>", "<<<END>>>"

def _ext_tog(r):
    """Extract prompt logprobs from Together completions response."""
    if "prompt" not in r: return None
    # Together returns prompt as a single dict, not list
    p = r["prompt"] if isinstance(r["prompt"], dict) else r["prompt"][0]
    if "logprobs" not in p: return None
    lp = []
    tk = p["logprobs"].get("tokens", [])
    tl = p["logprobs"].get("token_logprobs", [])
    for t, l in zip(tk, tl):
        lp.append({"token": t, "logprob": l})
    return lp

def _extract_label_prompt_tokens(raw: Dict[str, Any],
                                start: str = START, end: str = END) -> List[Dict[str, Any]]:
    """Extract tokens within the label markers from prompt logprobs."""
    try:
        # Try Together format first
        logprobs = _ext_tog(raw)
        if logprobs is None and "choices" in raw and raw["choices"]:
            choice = raw["choices"][0]
            
            # Try to get prompt logprobs (for echo=True responses)
            logprobs = None
            if "logprobs" in choice:
                if "prompt" in choice["logprobs"]:
                    logprobs = choice["logprobs"]["prompt"]
                elif "content" in choice["logprobs"]:
                    logprobs = choice["logprobs"]["content"]
                elif isinstance(choice["logprobs"], list):
                    logprobs = choice["logprobs"]
        
        if not logprobs:
            logger.warning("No prompt logprobs found in response")
            return []
        
        # Extract tokens within the label markers
        span, buf, on = [], "", False
        for item in logprobs:
            if isinstance(item, dict):
                tok = item.get("token", "")
            else:
                continue
                
            buf += tok
            if not on and start in buf:
                on = True
                buf = buf.split(start, 1)[1]
                continue
            if on:
                span.append(item)
                if end in buf:
                    break
        
        # Filter whitespace-only tokens for stability
        return [t for t in span if t.get("token", "").strip()]
    except Exception as e:
        logger.warning(f"Error extracting label tokens: {e}")
        return []

def _avg_margin(tokens: List[Dict[str, Any]]) -> float:
    """Calculate average margin between best token and alternatives."""
    margins = []
    for t in tokens:
        top_logprobs = t.get("top_logprobs", [])
        if isinstance(top_logprobs, list) and len(top_logprobs) > 0:
            chosen_token = t.get("token", "")
            chosen_logprob = t.get("logprob", 0)
            
            alts = []
            for alt in top_logprobs:
                if isinstance(alt, dict):
                    alt_token = alt.get("token", "")
                    if alt_token != chosen_token:
                        alts.append(alt.get("logprob", float("-inf")))
            
            if alts:
                margins.append(chosen_logprob - max(alts))
    
    return (sum(margins) / len(margins)) if margins else 0.0

@dataclass
class TFLLMetric:
    """
    Teacher-Forced Log-Likelihood (TFLL) metric for DSPy.
    One raw API call, no generation cost. Works with any optimizer.
    """
    raw_chat: Callable[..., Dict[str, Any]]
    model: str
    use_margin: bool = False
    margin_alpha: float = 0.5
    top_logprobs: int = 5
    allow_max_tokens_zero: bool = True
    fixed_system: str = "Be concise."
    
    def _messages_to_prompt(self, messages):
        """Convert messages to prompt."""
        parts = []
        for msg in messages:
            r = msg["role"]
            c = msg["content"]
            parts.append(f"{r.title()}: {c}")
        return "\n\n".join(parts)
    
    def _build_kwargs(self, msgs):
        if "together" in self.model.lower():
            p = self._messages_to_prompt(msgs)
            k = dict(model=self.model, prompt=p, echo=True, logprobs=1)
        else:
            k = dict(model=self.model, messages=msgs, echo=True, logprobs=True)
        k["temperature"] = 0
        return k
    
    def render_messages(self, instructions: str, x: str, y: str):
        """Render the messages for the LLM call with teacher-forced label."""
        return [
            {"role": "system", "content": f"{self.fixed_system}\n\n{instructions}".strip()},
            {"role": "user", "content": x},
            {"role": "assistant", "content": f"{START}{y}{END}"}
        ]
    
    def _score_once(self, messages) -> Dict[str, float]:
        """Score a single example with one raw API call."""
        kwargs = self._build_kwargs(messages)
        if self.use_margin:
            kwargs["top_logprobs"] = self.top_logprobs
        kwargs["max_tokens"] = 0 if self.allow_max_tokens_zero else 1
        
        try:
            raw = self.raw_chat(**kwargs)
            toks = _extract_label_prompt_tokens(raw)
            if not toks:
                logger.warning("No labeled tokens found")
                return {"tfll": float("-inf"), "margin": 0.0}
            
            tfll = sum(t.get("logprob", 0) for t in toks) / len(toks)
            margin = _avg_margin(toks) if self.use_margin else 0.0
            
            return {"tfll": tfll, "margin": margin}
        except Exception as e:
            logger.warning(f"Error calling raw_chat: {e}")
            return {"tfll": float("-inf"), "margin": 0.0}
    
    def __call__(self, example: Dict[str, Any], program) -> float:
        """Score an example given a program."""
        instr = None
        
        if hasattr(program, "signature") and program.signature:
            if hasattr(program.signature, "instructions"):
                instr = program.signature.instructions
        
        if instr is None and hasattr(program, "instructions"):
            instr = program.instructions
        
        if instr is None:
            instr = ""
        
        x = example.get("input", example.get("question", ""))
        y = example.get("label", example.get("answer", ""))
        
        msgs = self.render_messages(str(instr), str(x), str(y))
        res = self._score_once(msgs)
        
        score = res["tfll"] + (self.margin_alpha * res["margin"] if self.use_margin else 0.0)
        
        return score
    
    def feedback(self, example: Dict[str, Any], program) -> str:
        """Generate feedback text for GEPA optimization."""
        instr = None
        if hasattr(program, "signature") and program.signature:
            if hasattr(program.signature, "instructions"):
                instr = program.signature.instructions
        if instr is None and hasattr(program, "instructions"):
            instr = program.instructions
        if instr is None:
            instr = ""
        
        x = example.get("input", example.get("question", ""))
        y = example.get("label", example.get("answer", ""))
        
        msgs = self.render_messages(str(instr), str(x), str(y))
        
        kwargs = self._build_kwargs(msgs)
        if self.use_margin:
            kwargs["top_logprobs"] = self.top_logprobs
        kwargs["max_tokens"] = 0 if self.allow_max_tokens_zero else 1
        
        try:
            raw = self.raw_chat(**kwargs)
            toks = _extract_label_prompt_tokens(raw)
            if not toks:
                return "[TFLL] empty span"
            
            avg_lp = sum(t.get("logprob", 0) for t in toks) / len(toks)
            avg_mg = _avg_margin(toks) if self.use_margin else 0.0
            
            return f"[TFLL] avg_lp={avg_lp:.2f} nats/token, avg_margin={avg_mg:.2f}"
        except Exception as e:
            return f"[TFLL] error: {e}"
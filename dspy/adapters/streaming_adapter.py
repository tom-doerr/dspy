"""Low-latency streaming adapter — returns as soon as first output field is decisive.

Bypasses the normal non-streaming LM call path. Instead, sends a compact
prompt and streams the response, parsing output fields incrementally.
Returns immediately on complete first-field matches (e.g., "allow" needs
no reason, so we skip waiting for the second field).
"""

import json
import logging
import urllib.request
from typing import Any

from dspy.adapters.base import Adapter
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

# Compact prompt: pipe-delimited output, first token = first field
_PROMPT = (
    "{instructions}\n\n"
    "{field_desc}\n\n"
    "{input_block}\n\n"
    "Respond with fields separated by |, in order: {field_names}\n"
    "{field_hints}\n"
    "Output: "
)


class StreamingAdapter(Adapter):
    """Low-latency adapter: streams responses, parses incrementally,
    returns early when possible (e.g., "allow" needs no reason).

    Uses compact pipe-delimited format instead of field delimiters.
    """

    def __init__(self, early_return_fields=None, field_defaults=None,
                 max_tokens=20, timeout=10, **kwargs):
        super().__init__(**kwargs)
        self.early_return_fields = early_return_fields or {}
        self.field_defaults = field_defaults or {}
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _build_prompt(self, signature, inputs):
        fields = list(signature.output_fields.keys())
        hints = []
        for name, fi in signature.output_fields.items():
            extra = fi.json_schema_extra or {}
            desc = extra.get("desc", "")
            if desc:
                hints.append(f"  {name}: {desc}")
        inp = "\n".join(f"{k}: {v}" for k, v in inputs.items())
        return _PROMPT.format(
            instructions=signature.instructions.strip(),
            field_desc="", input_block=inp,
            field_names=", ".join(fields),
            field_hints="\n".join(hints))

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        prompt = self._build_prompt(signature, inputs)
        model = lm.model
        api_base = getattr(lm, "api_base", None) or lm.kwargs.get("api_base")
        extra = lm.kwargs.get("extra_body", {})
        body = {
            "model": model.split("/", 1)[-1] if "/" in model else model,
            "stream": True, "max_tokens": self.max_tokens,
            "temperature": lm_kwargs.get("temperature", 0.0),
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(extra)
        return self._stream_and_parse(api_base, body, signature)

    def _stream_tokens(self, api_base, body):
        data = json.dumps(body).encode()
        url = api_base.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for line in r:
                ln = line.decode().strip()
                if not ln.startswith("data: "):
                    continue
                raw = ln[6:]
                if raw == "[DONE]":
                    return
                d = json.loads(raw)
                t = d["choices"][0]["delta"].get("content", "")
                if t:
                    yield t

    def _stream_and_parse(self, api_base, body, signature):
        fields = list(signature.output_fields.keys())
        tokens = []
        for tok in self._stream_tokens(api_base, body):
            tokens.append(tok)
            result = self._try_parse(fields, tokens)
            if result is not None:
                return [result]
        result = self._try_parse(fields, tokens, final=True)
        if result is not None:
            return [result]
        text = "".join(tokens).strip()
        raise AdapterParseError(
            adapter_name="StreamingAdapter",
            signature=signature, lm_response=text,
            message=f"Unparseable: {text[:200]}")

    def _try_parse(self, fields, tokens, final=False):
        text = "".join(tokens).strip()
        parts = text.split("|", len(fields) - 1)
        p = {}
        for i, n in enumerate(fields):
            if i < len(parts):
                p[n] = parts[i].strip()
        if not p:
            return None
        er = self.early_return_fields.get(fields[0])
        if er and p.get(fields[0], "").lower() in er:
            for n in fields:
                p.setdefault(n, self.field_defaults.get(n, ""))
            return p
        if final or len(p) == len(fields):
            for n in fields:
                p.setdefault(n, self.field_defaults.get(n, ""))
            return p
        return None

    def parse(self, signature, completion):
        fields = list(signature.output_fields.keys())
        parts = completion.strip().split("|", len(fields) - 1)
        r = {}
        for i, n in enumerate(fields):
            r[n] = parts[i].strip() if i < len(parts) else self.field_defaults.get(n, "")
        return r

import pytest

import dspy


class DummyLM:
    def __init__(self, response: str):
        self.response = response
        self.prompt = None
        self.kwargs = None

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return [self.response]

    async def acall(self, prompt=None, messages=None, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return [self.response]


class BoolApproval(dspy.Signature):
    """Decide whether the tool call should be allowed."""

    policy: str = dspy.InputField()
    tool_name: str = dspy.InputField()
    tool_input: str = dspy.InputField()
    allow: bool = dspy.OutputField()


class ScoreOnly(dspy.Signature):
    """Extract a numeric score."""

    text: str = dspy.InputField()
    score: int = dspy.OutputField()


def test_raw_adapter_one_token_bool_defaults_to_one_generated_token():
    lm = DummyLM("1")
    adapter = dspy.RawAdapter(one_token=True)

    outputs = adapter(
        lm,
        lm_kwargs={},
        signature=BoolApproval,
        demos=[],
        inputs={
            "policy": "Only allow read-only tools.",
            "tool_name": "ReadFile",
            "tool_input": "/tmp/a.txt",
        },
    )

    assert outputs == [{"allow": True}]
    assert lm.kwargs["max_tokens"] == 1
    assert "Return exactly one token and nothing else." in lm.prompt


def test_raw_adapter_one_token_bool_parses_false_with_quotes_and_punctuation():
    lm = DummyLM("'0',")
    adapter = dspy.RawAdapter(one_token=True)

    outputs = adapter(
        lm,
        lm_kwargs={},
        signature=BoolApproval,
        demos=[],
        inputs={
            "policy": "Deny writes.",
            "tool_name": "WriteFile",
            "tool_input": "/tmp/a.txt",
        },
    )

    assert outputs == [{"allow": False}]


def test_raw_adapter_parses_single_scalar_output_without_chat_markers():
    lm = DummyLM("7")
    adapter = dspy.RawAdapter()

    outputs = adapter(
        lm,
        lm_kwargs={},
        signature=ScoreOnly,
        demos=[],
        inputs={"text": "Rate the urgency from 1 to 10."},
    )

    assert outputs == [{"score": 7}]


def test_raw_adapter_one_token_rejects_non_boolean_signatures():
    lm = DummyLM("1")
    adapter = dspy.RawAdapter(one_token=True)

    with pytest.raises(ValueError):
        adapter(
            lm,
            lm_kwargs={},
            signature=ScoreOnly,
            demos=[],
            inputs={"text": "hello"},
        )

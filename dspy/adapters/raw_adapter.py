import textwrap
import types
from typing import Any, Union, get_args, get_origin

from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import format_field_value, parse_value
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError


class RawAdapter(Adapter):
    """Low-overhead adapter for scalar prediction and classification tasks.

    `RawAdapter` renders a single flat prompt string instead of DSPy's standard
    chat-style structured prompt. It is meant for low-latency inference on local
    or OpenAI-compatible endpoints where the extra prompt and parsing overhead of
    `ChatAdapter` / `JSONAdapter` is undesirable.

    The special `one_token=True` mode is designed for signatures with exactly one
    boolean output field. In that mode, the adapter asks the model to emit only
    `true_token` for ``True`` or `false_token` for ``False`` and defaults the LM
    request to ``max_tokens=1``.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[type]] | None = None,
        *,
        one_token: bool = False,
        true_token: str = "1",
        false_token: str = "0",
        include_field_descriptions: bool = False,
        stop: list[str] | None = None,
    ):
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            native_response_types=native_response_types,
        )
        self.one_token = one_token
        self.true_token = true_token
        self.false_token = false_token
        self.include_field_descriptions = include_field_descriptions
        self.stop = stop

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        request_kwargs = dict(lm_kwargs)
        processed_signature = self._call_preprocess(lm, request_kwargs, signature, inputs)
        prompt = self.format(processed_signature, demos, inputs)
        request_kwargs = self._apply_generation_defaults(processed_signature, request_kwargs)
        outputs = lm(prompt=prompt, **request_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs, lm, request_kwargs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        request_kwargs = dict(lm_kwargs)
        processed_signature = self._call_preprocess(lm, request_kwargs, signature, inputs)
        prompt = self.format(processed_signature, demos, inputs)
        request_kwargs = self._apply_generation_defaults(processed_signature, request_kwargs)
        outputs = await lm.acall(prompt=prompt, **request_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs, lm, request_kwargs)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        instructions = textwrap.dedent(signature.instructions).strip()
        if instructions:
            parts.append(instructions)

        demo_blocks = [
            self._format_demo(signature, demo, index)
            for index, demo in enumerate(demos, start=1)
            if self._demo_has_required_fields(signature, demo)
        ]
        if demo_blocks:
            parts.append("Examples:\n\n" + "\n\n".join(demo_blocks))

        input_block = self._format_input_block(signature, inputs)
        if input_block:
            parts.append("Input:\n" + input_block)

        parts.append(self._format_output_contract(signature))
        return "\n\n".join(part for part in parts if part).strip()

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        if self.one_token:
            name, field = self._get_single_output_field(signature)
            try:
                token = self._extract_single_token(completion)
                return {name: parse_value(token, field.annotation)}
            except Exception as exc:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse one-token output for field {name!r}: {exc}",
                ) from exc

        if len(signature.output_fields) == 1:
            name, field = self._get_single_output_field(signature)
            value_text = completion.strip()
            if not value_text:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    message="The LM returned an empty response.",
                )
            try:
                return {name: parse_value(value_text, field.annotation)}
            except Exception as exc:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse output field {name!r}: {exc}",
                ) from exc

        raw_values = self._parse_key_value_lines(signature, completion)
        parsed: dict[str, Any] = {}
        for name, field in signature.output_fields.items():
            if name not in raw_values:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    parsed_result=raw_values,
                    message=f"Missing output field {name!r} in raw response.",
                )
            try:
                parsed[name] = parse_value(raw_values[name], field.annotation)
            except Exception as exc:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    parsed_result=raw_values,
                    message=f"Failed to parse output field {name!r}: {exc}",
                ) from exc
        return parsed

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        return self._format_output_block(signature, outputs, missing_field_message=missing_field_message)

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, str]:
        return {
            "prompt": self.format(signature=signature, demos=demos, inputs=inputs),
            "completion": self._format_output_block(signature, outputs),
        }

    def _apply_generation_defaults(
        self,
        signature: type[Signature],
        lm_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        request_kwargs = dict(lm_kwargs)
        if self.one_token:
            self._validate_one_token_signature(signature)
            request_kwargs.setdefault("max_tokens", 1)
            if self.stop is not None:
                request_kwargs.setdefault("stop", self.stop)
        elif self.stop is not None:
            request_kwargs.setdefault("stop", self.stop)
        return request_kwargs

    def _format_demo(self, signature: type[Signature], demo: dict[str, Any], index: int) -> str:
        demo_inputs = {name: demo[name] for name in signature.input_fields if name in demo and demo[name] is not None}
        demo_outputs = {name: demo[name] for name in signature.output_fields if name in demo and demo[name] is not None}

        parts = [f"Example {index}"]
        if demo_inputs:
            parts.append("Input:\n" + self._format_input_block(signature, demo_inputs))
        if demo_outputs:
            parts.append("Output:\n" + self._format_output_block(signature, demo_outputs))
        return "\n\n".join(parts)

    def _format_input_block(self, signature: type[Signature], values: dict[str, Any]) -> str:
        lines: list[str] = []
        for name, field in signature.input_fields.items():
            if name not in values:
                continue
            description = self._field_description(name, field)
            if description:
                lines.append(f"# {name}: {description}")
            lines.append(f"{name}:")
            lines.append(str(format_field_value(field_info=field, value=values[name])))
        return "\n".join(lines).strip()

    def _format_output_contract(self, signature: type[Signature]) -> str:
        if self.one_token:
            name, field = self._get_single_output_field(signature)
            self._validate_one_token_signature(signature)
            description = self._field_description(name, field)
            lines = []
            if description:
                lines.append(f"Target output `{name}`: {description}")
            lines.extend(
                [
                    "Return exactly one token and nothing else.",
                    f"{self.true_token} = True",
                    f"{self.false_token} = False",
                    "No explanation. No punctuation. No extra text.",
                ]
            )
            return "\n".join(lines)

        if len(signature.output_fields) == 1:
            name, field = self._get_single_output_field(signature)
            description = self._field_description(name, field)
            if description:
                return f"Return only the value for `{name}` ({description}). No extra text."
            return f"Return only the value for `{name}`. No extra text."

        lines = ["Return only the output fields below, one per line, with no extra text:"]
        for name, field in signature.output_fields.items():
            description = self._field_description(name, field)
            if description:
                lines.append(f"{name}=  # {description}")
            else:
                lines.append(f"{name}=")
        return "\n".join(lines)

    def _format_output_block(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        *,
        missing_field_message: str | None = None,
    ) -> str:
        if self.one_token:
            name, _ = self._get_single_output_field(signature)
            value = outputs.get(name, missing_field_message)
            if value in (None, missing_field_message):
                return str(missing_field_message or "")
            bool_value = parse_value(value, bool)
            return self._serialize_boolean_output(bool_value)

        if len(signature.output_fields) == 1:
            name, field = self._get_single_output_field(signature)
            value = outputs.get(name, missing_field_message)
            if value is None:
                value = missing_field_message
            return str(format_field_value(field_info=field, value=value))

        lines: list[str] = []
        for name, field in signature.output_fields.items():
            value = outputs.get(name, missing_field_message)
            if value is None:
                value = missing_field_message
            rendered_value = str(format_field_value(field_info=field, value=value))
            value_lines = rendered_value.splitlines() or [""]
            first_line, *rest = value_lines
            lines.append(f"{name}={first_line}")
            lines.extend(rest)
        return "\n".join(lines).strip()

    def _parse_key_value_lines(self, signature: type[Signature], completion: str) -> dict[str, str]:
        expected_fields = list(signature.output_fields.keys())
        current_field: str | None = None
        current_lines: list[str] = []
        parsed: dict[str, str] = {}

        for line in completion.splitlines():
            if "=" in line:
                key, remainder = line.split("=", 1)
                key = key.strip()
                if key in signature.output_fields:
                    if current_field is not None and current_field not in parsed:
                        parsed[current_field] = "\n".join(current_lines).strip()
                    current_field = key
                    current_lines = [remainder.strip()]
                    continue
            if current_field is not None:
                current_lines.append(line)

        if current_field is not None and current_field not in parsed:
            parsed[current_field] = "\n".join(current_lines).strip()

        return {name: parsed[name] for name in expected_fields if name in parsed}

    def _extract_single_token(self, completion: str) -> str:
        token = completion.strip()
        if not token:
            raise ValueError("The LM returned an empty response for one-token mode.")

        token = token.split()[0]
        token = token.strip("\"'")
        token = token.lstrip(":=(")
        token = token.rstrip(",.;:!?)]}\"'")
        return token

    def _validate_one_token_signature(self, signature: type[Signature]) -> None:
        _, field = self._get_single_output_field(signature)
        annotation = self._unwrap_optional(field.annotation)
        if annotation is not bool:
            raise ValueError(
                "RawAdapter(one_token=True) currently requires exactly one boolean output field. "
                f"Received annotation {field.annotation!r}."
            )
        if any(ch.isspace() for ch in self.true_token) or any(ch.isspace() for ch in self.false_token):
            raise ValueError("true_token and false_token must not contain whitespace in one-token mode.")
        if self.true_token == self.false_token:
            raise ValueError("true_token and false_token must be different values.")

    def _get_single_output_field(self, signature: type[Signature]) -> tuple[str, FieldInfo]:
        if len(signature.output_fields) != 1:
            raise ValueError(
                f"RawAdapter requires exactly one output field for this mode, but received {len(signature.output_fields)}."
            )
        return next(iter(signature.output_fields.items()))

    def _field_description(self, name: str, field: FieldInfo) -> str:
        if not self.include_field_descriptions:
            return ""
        extra = field.json_schema_extra or {}
        desc = extra.get("desc", "")
        if desc == f"${{{name}}}":
            return ""
        return str(desc).strip()

    def _serialize_boolean_output(self, value: bool) -> str:
        return self.true_token if value else self.false_token

    def _demo_has_required_fields(self, signature: type[Signature], demo: dict[str, Any]) -> bool:
        has_input = any(name in demo and demo[name] is not None for name in signature.input_fields)
        has_output = any(name in demo and demo[name] is not None for name in signature.output_fields)
        return has_input and has_output

    def _unwrap_optional(self, annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin in (Union, types.UnionType):
            non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(non_none_args) == 1:
                return non_none_args[0]
        return annotation

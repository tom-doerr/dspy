import logging
import uuid
from typing import Any

import dspy
from dspy.primitives import Example, Module
from dspy.utils.callback import ACTIVE_CALL_ID, with_callbacks

logger = logging.getLogger(__name__)


class Teleprompter:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        compile_fn = cls.__dict__.get("compile")
        if compile_fn is None or getattr(compile_fn, "__dspy_callbacks_wrapped__", False):
            return
        wrapped = with_callbacks(compile_fn)
        wrapped.__dspy_callbacks_wrapped__ = True
        cls.compile = wrapped

    def compile(self, student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None, **kwargs) -> Module:
        """
        Optimize the student program.

        Args:
            student: The student program to optimize.
            trainset: The training set to use for optimization.
            teacher: The teacher program to use for optimization.
            valset: The validation set to use for optimization.

        Returns:
            The optimized student program.
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of the teleprompter.

        Returns:
            The parameters of the teleprompter.
        """
        return self.__dict__

    def _emit_progress(self, **event) -> None:
        callbacks = dspy.settings.get("callbacks", []) + getattr(self, "callbacks", [])
        if not callbacks:
            return

        call_id = ACTIVE_CALL_ID.get() or uuid.uuid4().hex
        for callback in callbacks:
            try:
                callback.on_teleprompter_progress(
                    call_id=call_id,
                    instance=self,
                    event=event,
                )
            except Exception as exc:
                logger.warning("Error when calling teleprompter progress callback %s: %s", callback, exc)

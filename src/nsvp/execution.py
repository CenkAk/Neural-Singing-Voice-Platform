from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionControl:
    check_cancelled: Callable[[], None]
    process_started: Callable[[int, float], None]
    process_finished: Callable[[int], None]
    log_directory: Path | None = None


current_execution: ContextVar[ExecutionControl | None] = ContextVar("execution", default=None)


def check_cancelled() -> None:
    control = current_execution.get()
    if control is not None:
        control.check_cancelled()


@contextmanager
def execution_scope(control: ExecutionControl) -> Iterator[None]:
    token = current_execution.set(control)
    try:
        yield
    finally:
        current_execution.reset(token)

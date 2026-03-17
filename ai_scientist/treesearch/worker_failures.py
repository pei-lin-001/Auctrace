from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import openai

from .journal import Node
from .utils.metric import WorstMetricValue

LLM_TRANSPORT_FAILURE = "LLMTransportFailure"


@dataclass(frozen=True)
class SerializedWorkerError:
    error_type: str
    message: str
    traceback_text: str
    stage_name: str | None


def is_retryable_llm_exception(exc: Exception) -> bool:
    retryable_types = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
    return isinstance(exc, retryable_types)


def is_infra_failure_node(node: Node) -> bool:
    return node.exc_type == LLM_TRANSPORT_FAILURE


def serialize_worker_error(
    exc: Exception,
    traceback_text: str,
    stage_name: str | None,
) -> SerializedWorkerError:
    return SerializedWorkerError(
        error_type=type(exc).__name__,
        message=str(exc),
        traceback_text=traceback_text,
        stage_name=stage_name,
    )


def build_retryable_llm_failure_node(
    parent_node: Node | None,
    stage_name: str | None,
    exc: Exception,
    traceback_text: str,
) -> dict[str, Any]:
    error = serialize_worker_error(exc, traceback_text, stage_name)
    summary = (
        "LLM request failed before experiment execution. "
        f"stage={error.stage_name}; "
        f"error={error.error_type}: {error.message}"
    )
    node = Node(
        plan="Retryable LLM transport failure while generating the next node.",
        code=parent_node.code if parent_node else "",
        parent=parent_node,
        _term_out=[error.traceback_text],
        exc_type=LLM_TRANSPORT_FAILURE,
        exc_info={
            "original_type": error.error_type,
            "message": error.message,
            "stage_name": error.stage_name,
        },
        exc_stack=None,
        analysis=summary,
        metric=WorstMetricValue(),
        is_buggy=True,
        is_buggy_plots=None,
    )
    return node.to_dict()

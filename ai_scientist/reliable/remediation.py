from __future__ import annotations

import json
import os
import os.path as osp
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import httpx
import openai

from .claim_ledger import ClaimLedgerError
from .claim_traceability import ClaimTraceabilityError
from .errors import (
    FactStoreFormatError,
    InvalidFactKeyError,
    InvalidFigurePathError,
    InvalidParamKeyError,
    ParamStoreFormatError,
    ReliableWriteupError,
    SymbolicLatexError,
    UnknownFactKeyError,
    UnknownParamKeyError,
)
from .latex_compile import LatexCommandFailure
from .outsider_audit import OutsiderAuditError

REMEDIATION_REPORT_SCHEMA = "auctrace.remediation_report.v1"


@dataclass(frozen=True)
class RemediationDecision:
    schema: str
    phase: str
    failure_code: str
    blocking: bool
    retryable: bool
    retry_target: str
    exception_type: str
    message: str
    traceback: str


def _target_guidance(retry_target: str) -> str:
    if retry_target == "writeup":
        return "Repair the LaTeX manuscript directly. Keep fact placeholders valid and remove unsupported claims."
    if retry_target == "claim_ledger":
        return (
            "Repair claim bindings. Every used fact key must be covered by claims, "
            "and key-backed claims must keep clear artifact and location bindings."
        )
    if retry_target == "artifact_manifest":
        return (
            "Repair artifact bindings. Only reference figure/table identities that can be "
            "resolved from the artifact manifest."
        )
    if retry_target == "outsider_audit":
        return (
            "Repair unsupported or unclear narrative claims so the outsider audit can verify "
            "them from the provided facts and artifacts."
        )
    if retry_target == "fact_store":
        return "Do not retry writeup until the upstream fact store inconsistency is repaired."
    return "Repair the previous blocking failure before continuing."


def _failure_specific_guidance(decision: RemediationDecision) -> str:
    message = decision.message.lower()
    if decision.failure_code == "LLMConnectionError":
        return (
            "- Keep the latest valid LaTeX draft and retry the same repair goal after the connection recovers.\n"
            "- Do not introduce new claims or figure paths while recovering from a transient API/network failure.\n"
        )
    if decision.failure_code == "UnknownFactKeyError":
        return (
            "- Replace only the unknown \\fact{KEY} placeholders using the exact suggested keys from the error.\n"
            "- Do not delete all fact placeholders; the revised manuscript must still contain at least one valid \\fact{KEY}.\n"
        )
    if decision.failure_code == "UnknownParamKeyError":
        return (
            "- Replace only the unknown \\param{KEY} placeholders using the exact suggested keys from the error.\n"
            "- If no exact param key exists, rewrite the sentence qualitatively instead of inventing a new constant.\n"
        )
    if decision.failure_code == "InvalidFigurePathError":
        return (
            "- Replace every invalid \\includegraphics target using only exact strings from artifact_manifest_summary.includegraphics_targets.\n"
            "- Do not reference stale experiment_results PNG paths or figures from older runs.\n"
        )
    if decision.failure_code == "LatexCommandFailure":
        return (
            "- Repair compile-breaking LaTeX directly.\n"
            "- Keep the scaffold before \\end{filecontents} unchanged; do not rewrite package imports, theorem declarations, or the references.bib scaffold.\n"
            "- Only use figure paths from the current artifact manifest; remove stale experiment_results PNG paths.\n"
        )
    if (
        decision.failure_code == "SymbolicLatexError"
        and "expected at least 1 unique \\\\fact{key} usage" in message
    ):
        return (
            "- Re-introduce at least one valid \\fact{KEY} placeholder from writeup_context_pack.preferred_fact_refs or WRITEUP_SYMBOLIC_SUMMARY.*.fact_refs.\n"
            "- Do not replace all quantitative findings with qualitative prose; anchor at least one main-text result sentence to an exact fact key.\n"
        )
    if (
        decision.failure_code == "SymbolicLatexError"
        and "unanchored numeric literals" in message
    ):
        return (
            "- Move experimental values back to exact \\fact{KEY} placeholders.\n"
            "- Move method/setup/constants to exact \\param{KEY} placeholders when an exact param key exists; otherwise rewrite qualitatively.\n"
            "- Do not write approximate relative reductions or percentage ranges such as 'over 30%' or '15-25%'; either anchor them to facts or rewrite qualitatively.\n"
        )
    return ""


def classify_remediation_failure(
    *,
    phase: str,
    exc: Exception,
    traceback_text: str,
) -> RemediationDecision:
    failure_code = "UnhandledReliableFailure"
    retryable = False
    retry_target = "stop"
    exc_name = exc.__class__.__name__
    if exc_name in {"APIConnectionError", "APITimeoutError"} or isinstance(
        exc,
        (
            httpx.TransportError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        failure_code = "LLMConnectionError"
        retryable = True
        retry_target = "writeup"
    if isinstance(exc, UnknownFactKeyError):
        failure_code = "UnknownFactKeyError"
        retryable = True
        retry_target = "writeup"
    elif isinstance(exc, UnknownParamKeyError):
        failure_code = "UnknownParamKeyError"
        retryable = True
        retry_target = "writeup"
    elif isinstance(exc, InvalidFigurePathError):
        failure_code = "InvalidFigurePathError"
        retryable = True
        retry_target = "writeup"
    elif isinstance(exc, (FactStoreFormatError, InvalidFactKeyError, ParamStoreFormatError, InvalidParamKeyError)):
        failure_code = "FactStoreSyncError"
        retry_target = "fact_store"
    elif isinstance(exc, SymbolicLatexError):
        failure_code = "SymbolicLatexError"
        retryable = True
        retry_target = "writeup"
    elif isinstance(exc, (ClaimLedgerError, ClaimTraceabilityError)):
        failure_code = "ClaimCoverageError"
        retryable = True
        retry_target = "claim_ledger"
    elif isinstance(exc, OutsiderAuditError):
        failure_code = "OutsiderAuditBlockingIssue"
        retryable = True
        retry_target = "outsider_audit"
    elif isinstance(exc, LatexCommandFailure):
        failure_code = "LatexCommandFailure"
        retryable = True
        retry_target = "writeup"
    elif isinstance(exc, FileNotFoundError):
        failure_code = "ArtifactBindingError"
        retryable = True
        retry_target = "artifact_manifest"
    elif isinstance(exc, ReliableWriteupError):
        failure_code = exc.__class__.__name__
        retry_target = "writeup"
    return RemediationDecision(
        schema=REMEDIATION_REPORT_SCHEMA,
        phase=phase,
        failure_code=failure_code,
        blocking=True,
        retryable=retryable,
        retry_target=retry_target,
        exception_type=exc.__class__.__name__,
        message=str(exc),
        traceback=traceback_text,
    )


def save_remediation_report(path: str, decision: RemediationDecision) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(decision), f, indent=2, ensure_ascii=True)


def load_remediation_report(path: str) -> RemediationDecision | None:
    if not osp.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid remediation report payload at {path}")
    if str(payload.get("schema") or "").strip() != REMEDIATION_REPORT_SCHEMA:
        raise ValueError(f"Unexpected remediation schema at {path}")
    return RemediationDecision(
        schema=REMEDIATION_REPORT_SCHEMA,
        phase=str(payload.get("phase") or "").strip(),
        failure_code=str(payload.get("failure_code") or "").strip(),
        blocking=bool(payload.get("blocking", True)),
        retryable=bool(payload.get("retryable", False)),
        retry_target=str(payload.get("retry_target") or "stop").strip(),
        exception_type=str(payload.get("exception_type") or "").strip(),
        message=str(payload.get("message") or "").strip(),
        traceback=str(payload.get("traceback") or "").strip(),
    )


def build_remediation_prompt_block(context: Mapping[str, Any] | RemediationDecision | None) -> str:
    if context is None:
        return ""
    if isinstance(context, RemediationDecision):
        decision = context
    else:
        decision = RemediationDecision(
            schema=str(context.get("schema") or REMEDIATION_REPORT_SCHEMA),
            phase=str(context.get("phase") or "").strip(),
            failure_code=str(context.get("failure_code") or "").strip(),
            blocking=bool(context.get("blocking", True)),
            retryable=bool(context.get("retryable", False)),
            retry_target=str(context.get("retry_target") or "stop").strip(),
            exception_type=str(context.get("exception_type") or "").strip(),
            message=str(context.get("message") or "").strip(),
            traceback=str(context.get("traceback") or "").strip(),
        )
    format_contract = ""
    if decision.retry_target == "writeup":
        format_contract = "- Output format: return exactly one ```latex fenced block with the full template.tex.\n"
    return (
        "PREVIOUS ATTEMPT FAILURE SUMMARY:\n"
        f"- failure_code: {decision.failure_code}\n"
        f"- retry_target: {decision.retry_target}\n"
        f"- exception_type: {decision.exception_type}\n"
        f"- message: {decision.message}\n"
        f"- repair_focus: {_target_guidance(decision.retry_target)}\n"
        f"{_failure_specific_guidance(decision)}"
        "- This retry must explicitly fix the blocking issue above instead of repeating the same draft.\n"
        f"{format_contract}"
    )


def should_retry_writeup(decision: RemediationDecision | None) -> bool:
    if decision is None:
        return False
    return decision.retryable and decision.retry_target != "fact_store"


def remediation_retry_target(
    context: Mapping[str, Any] | RemediationDecision | None,
) -> str:
    if context is None:
        return ""
    if isinstance(context, RemediationDecision):
        return context.retry_target
    return str(context.get("retry_target") or "").strip()


def should_reuse_symbolic_writeup_artifacts(
    context: Mapping[str, Any] | RemediationDecision | None,
) -> bool:
    return remediation_retry_target(context) in {
        "claim_ledger",
        "outsider_audit",
        "artifact_manifest",
    }


def remediation_retry_banner(decision: RemediationDecision | None) -> str:
    if decision is None:
        return "[remediation] no remediation report available."
    return (
        "[remediation] retrying with targeted repair: "
        f"failure_code={decision.failure_code}, retry_target={decision.retry_target}"
    )


def print_remediation_report(decision: RemediationDecision) -> None:
    print("[remediation] failure summary:")
    print(json.dumps(asdict(decision), indent=2, ensure_ascii=True))

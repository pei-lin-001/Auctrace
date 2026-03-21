from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_scientist.reliable.metrics import (
    compute_ceb,
    compute_eru,
    compute_ics_auto,
    compute_nas_from_latex,
    compute_ptc,
    overlap_numeric_claims_for_nia,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single experiment run under the MAC (Manuscript-Artifact Consistency) protocol."
    )
    parser.add_argument(
        "--idea-dir",
        required=True,
        help="Path to experiments/<run_dir>.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/0-run",
        help="Relative path from idea-dir to log dir (default: logs/0-run).",
    )
    parser.add_argument(
        "--out",
        default="evaluation_report.json",
        help="Output filename under idea-dir (default: evaluation_report.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    idea_dir = Path(args.idea_dir).expanduser().resolve()
    if not idea_dir.exists():
        raise FileNotFoundError(f"idea-dir does not exist: {idea_dir}")

    latex_dir = idea_dir / "latex"
    log_dir = (idea_dir / args.log_dir).resolve()

    template_tex = _read_text(latex_dir / "template.tex")
    rendered_tex = _read_text(latex_dir / "template.rendered.tex")

    claim_ledger = _read_json(latex_dir / "claim_ledger.json")
    claim_trace_index = _read_json(latex_dir / "claim_trace_index.json")
    used_facts = _read_json(latex_dir / "used_facts.json")
    outsider_audit = _read_json(latex_dir / "outsider_audit.json")

    fact_store = _read_json(log_dir / "fact_store.json")
    artifact_manifest = _read_json(log_dir / "artifact_manifest.json")

    report: dict[str, Any] = {
        "schema": "auctrace.mac_evaluation.v1",
        "idea_dir": str(idea_dir),
        "log_dir": str(log_dir),
        "latex_dir": str(latex_dir),
        "artifacts_present": {
            "template_tex": template_tex is not None,
            "template_rendered_tex": rendered_tex is not None,
            "claim_ledger": claim_ledger is not None,
            "claim_trace_index": claim_trace_index is not None,
            "used_facts": used_facts is not None,
            "outsider_audit": outsider_audit is not None,
            "fact_store": fact_store is not None,
            "artifact_manifest": artifact_manifest is not None,
        },
        "metrics": {},
    }

    metrics: dict[str, Any] = {}

    if template_tex is not None and rendered_tex is not None:
        metrics["NAS"] = compute_nas_from_latex(
            symbolic_tex=template_tex,
            rendered_tex=rendered_tex,
        )

    if fact_store is not None and used_facts is not None:
        metrics["ERU"] = compute_eru(
            fact_store=fact_store,
            used_facts=used_facts,
        )

    if outsider_audit is not None:
        metrics["ICS_auto"] = compute_ics_auto(outsider_audit)

    if (
        used_facts is not None
        and claim_trace_index is not None
        and artifact_manifest is not None
    ):
        metrics["PTC"] = compute_ptc(
            used_facts=used_facts,
            claim_trace_index=claim_trace_index,
            artifact_manifest=artifact_manifest,
        )

    if claim_ledger is not None:
        if fact_store is not None or artifact_manifest is not None or claim_trace_index is not None:
            metrics["CEB"] = compute_ceb(
                claim_ledger=claim_ledger,
                fact_store=fact_store,
                artifact_manifest=artifact_manifest,
                claim_trace_index=claim_trace_index,
            )
        metrics["overlap"] = overlap_numeric_claims_for_nia(claim_ledger)

    report["metrics"] = metrics

    out_path = idea_dir / args.out
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    # Human-readable summary
    print(f"[evaluate] wrote: {out_path}")
    if "NAS" in metrics:
        nas = metrics["NAS"].get("NAS")
        print(f"[evaluate] NAS={nas} (T_result={metrics['NAS'].get('T_result_count')})")
    if "ERU" in metrics:
        eru = metrics["ERU"].get("ERU")
        print(f"[evaluate] ERU={eru} ({metrics['ERU'].get('used_keys')}/{metrics['ERU'].get('total_keys')})")
    if "CEB" in metrics:
        ceb = metrics["CEB"].get("CEB")
        print(f"[evaluate] CEB={ceb} ({metrics['CEB'].get('bound_claims')}/{metrics['CEB'].get('total_claims')})")
    if "PTC" in metrics:
        ptc = metrics["PTC"].get("PTC")
        print(f"[evaluate] PTC={ptc} ({metrics['PTC'].get('covered_keys')}/{metrics['PTC'].get('used_keys')})")
    if "ICS_auto" in metrics:
        ics = metrics["ICS_auto"].get("ICS_auto")
        print(f"[evaluate] ICS_auto={ics} (weighted={metrics['ICS_auto'].get('weighted_issues')})")
    if "overlap" in metrics:
        overlap = metrics["overlap"].get("overlap_numeric_comparative_count")
        print(f"[evaluate] overlap_numeric_comparative={overlap}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os.path as osp
import re
from typing import Any, Dict

from .errors import SymbolicLatexError, UnknownFactKeyError
from .facts import FactStore, format_fact_value_for_latex
from .params import ParamStore, format_param_value_for_latex
from .placeholders import list_fact_keys_in_order, list_param_keys_in_order

_FACT_MACRO_RE = re.compile(r"\\fact\{([^{}]+)\}")
_PARAM_MACRO_RE = re.compile(r"\\param\{([^{}]+)\}")


def render_symbolic_latex(
    symbolic_tex: str,
    store: FactStore,
    param_store: ParamStore | None = None,
) -> tuple[str, Dict[str, Any]]:
    keys = list_fact_keys_in_order(symbolic_tex)
    param_keys = list_param_keys_in_order(symbolic_tex)
    used: Dict[str, Any] = {
        "schema": "auctrace.used_facts.v1",
        "used_keys": [],
        "facts": [],
        "used_params": [],
        "params": [],
    }

    def _replace_fact(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        record = store.get(key)  # raises UnknownFactKeyError
        return format_fact_value_for_latex(record)

    def _replace_param(match: re.Match[str]) -> str:
        if param_store is None:
            raise SymbolicLatexError(
                "Rendered LaTeX contains \\param{...} placeholders but no ParamStore was provided."
            )
        key = match.group(1).strip()
        record = param_store.get(key)
        return format_param_value_for_latex(record)

    try:
        rendered = _FACT_MACRO_RE.sub(_replace_fact, symbolic_tex)
        rendered = _PARAM_MACRO_RE.sub(_replace_param, rendered)
    except UnknownFactKeyError:
        raise
    except Exception as e:
        raise SymbolicLatexError(f"Failed to render symbolic LaTeX: {e}") from e

    for key in keys:
        rec = store.get(key)
        used["used_keys"].append(key)
        used["facts"].append(
            {
                "key": rec.key,
                "meaning": rec.meaning,
                "format": rec.format,
                "provenance": rec.provenance,
            }
        )
    if param_store is not None:
        for key in param_keys:
            rec = param_store.get(key)
            used["used_params"].append(key)
            used["params"].append(
                {
                    "key": rec.key,
                    "meaning": rec.meaning,
                    "format": rec.format,
                    "provenance": rec.provenance,
                }
            )

    if "\\fact{" in rendered:
        # This should be impossible if regex replacement ran, but keep it explicit.
        raise SymbolicLatexError("Rendered LaTeX still contains unresolved \\fact{...} placeholders.")
    if "\\param{" in rendered:
        raise SymbolicLatexError("Rendered LaTeX still contains unresolved \\param{...} placeholders.")

    return rendered, used


def render_symbolic_tex_file(
    *,
    symbolic_tex_path: str,
    fact_store_path: str,
    param_store_path: str | None,
    rendered_tex_path: str,
    used_facts_path: str | None,
) -> None:
    store = FactStore.load_json(fact_store_path)
    param_store = ParamStore.load_json(param_store_path) if param_store_path else None
    with open(symbolic_tex_path, "r", encoding="utf-8") as f:
        symbolic_tex = f.read()

    rendered_tex, used = render_symbolic_latex(symbolic_tex, store, param_store)

    parent = osp.dirname(rendered_tex_path)
    if parent:
        import os

        os.makedirs(parent, exist_ok=True)
    with open(rendered_tex_path, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    if used_facts_path is not None:
        with open(used_facts_path, "w", encoding="utf-8") as f:
            json.dump(used, f, indent=2, ensure_ascii=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render symbolic LaTeX (\\fact{key}) using a FactStore JSON.")
    parser.add_argument("--symbolic-tex", required=True, help="Path to symbolic template.tex (contains \\fact{...}).")
    parser.add_argument("--fact-store", required=True, help="Path to fact_store.json produced by the pipeline.")
    parser.add_argument("--param-store", default="", help="Optional param_store.json path for \\param{...} placeholders.")
    parser.add_argument("--out-tex", required=True, help="Output rendered .tex path.")
    parser.add_argument("--used-facts", default="", help="Optional output used_facts.json path (empty disables).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    used_path = args.used_facts.strip() or None
    render_symbolic_tex_file(
        symbolic_tex_path=args.symbolic_tex,
        fact_store_path=args.fact_store,
        param_store_path=args.param_store.strip() or None,
        rendered_tex_path=args.out_tex,
        used_facts_path=used_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render README.md from README.template.md and results/summary.json so no number is typed by hand.

Tokens look like {{decisions.accuracy}}. An unknown or malformed token is an error, and so is a summary that
no longer supports a directional claim the template makes (e.g. "lower p50 latency").
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TOKEN = re.compile(r"\{\{\s*([a-z0-9_.]+)\s*\}\}")

METHODS = {
    "decisions": "decisions-gpt-6-luna",
    "responses": "responses-gpt-6-luna",
    "nano": "responses-gpt-5.4-nano",
}


class RenderError(Exception):
    pass


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _ms(seconds: float) -> str:
    return f"{1000 * seconds:.0f}"


def _usd(x: float) -> str:
    return f"${x:.4f}"


def _row(s: dict[str, Any], suffix: str = "") -> dict[str, str]:
    return {
        "accuracy": _pct(s["accuracy"]),
        "ece": _num(s["ece" + suffix]),
        "aurc": _num(s["aurc" + suffix], 4),
        "threshold": _num(s["dev_threshold" + suffix]),
        "coverage": _pct(s["test_coverage_at_threshold" + suffix]),
        "coverage_accuracy": _pct(s["test_accuracy_at_threshold" + suffix]),
        "p50_ms": _ms(s["latency_p50_s"]),
        "p95_ms": _ms(s["latency_p95_s"]),
        "cost_per_1k": _usd(s["cost_per_1k_usd"]),
        "n": str(s["n"]),
        "n_dev": str(s["n_dev"]),
        "n_test": str(s["n_test"]),
    }


def check_claims(summary: dict[str, Any]) -> None:
    m = summary["methods"]
    d, r = m[METHODS["decisions"]], m[METHODS["responses"]]
    claims = [
        (d["latency_p50_s"] < r["latency_p50_s"], "Decisions p50 latency is lower than Responses (same model)"),
        (d["cost_per_1k_usd"] < r["cost_per_1k_usd"], "Decisions cost per 1k is lower than Responses (same model)"),
        (d["temperature"] < 1.0, "raw Decisions probabilities are under-confident (fitted T < 1)"),
        (d["ece_calibrated"] < d["ece"], "temperature scaling lowers Decisions ECE"),
    ]
    broken = [text for ok, text in claims if not ok]
    if broken:
        raise RenderError("README.template.md claims no longer hold; rewrite them: " + "; ".join(broken))


def build_context(summary: dict[str, Any]) -> dict[str, str]:
    try:
        m = {alias: summary["methods"][name] for alias, name in METHODS.items()}
        d, r = m["decisions"], m["responses"]
        check_claims(summary)
        ctx: dict[str, str] = {}
        for alias, s in m.items():
            ctx.update({f"{alias}.{k}": v for k, v in _row(s).items()})
        ctx.update({f"decisions_ts.{k}": v for k, v in _row(d, "_calibrated").items()})
        ctx["decisions.temperature"] = _num(d["temperature"], 2)
        ctx["decisions.ece_calibrated"] = _num(d["ece_calibrated"])
        ctx["headline.latency_ratio"] = f"{r['latency_p50_s'] / d['latency_p50_s']:.1f}"
        ctx["headline.cost_reduction_pct"] = f"{100 * (1 - d['cost_per_1k_usd'] / r['cost_per_1k_usd']):.0f}%"
        ctx["target"] = f"{100 * summary['target_accuracy']:.0f}%"
        ctx["total_spend"] = _usd(summary["total_spend_usd"])
    except KeyError as e:
        raise RenderError(f"summary.json is missing {e}; re-run python -m decision_triage.report") from e
    return ctx


def render(template: str, ctx: dict[str, str]) -> str:
    unknown = sorted({t for t in TOKEN.findall(template) if t not in ctx})
    if unknown:
        raise RenderError("unknown README tokens: " + ", ".join(unknown))
    out = TOKEN.sub(lambda mt: ctx[mt.group(1)], template)
    leftover = re.findall(r"\{\{.*?\}\}|\{\{|\}\}", out)
    if leftover:
        raise RenderError(f"malformed README tokens: {leftover}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--template", type=Path, default=Path("README.template.md"))
    p.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    p.add_argument("--out", type=Path, default=Path("README.md"))
    p.add_argument("--check", action="store_true", help="exit 1 if --out is not what the template renders to")
    args = p.parse_args(argv)
    try:
        text = render(args.template.read_text(), build_context(json.loads(args.summary.read_text())))
    except RenderError as e:
        print(f"render_readme: {e}", file=sys.stderr)
        return 2
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            print(f"render_readme: {args.out} is stale; run python scripts/render_readme.py", file=sys.stderr)
            return 1
        return 0
    args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

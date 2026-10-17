import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from decision_triage import calibrate, metrics  # noqa: E402
from decision_triage.run_eval import load_jsonl  # noqa: E402
from decision_triage.taxonomy import QUEUE_NAMES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
CALIBRATED_SUFFIX = " (temperature-scaled on dev)"
CALIBRATED_KEYS = ("ece", "aurc", "dev_threshold", "test_coverage_at_threshold", "test_accuracy_at_threshold")


def _arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "confidence": np.array([r["confidence"] for r in rows], dtype=float),
        "correct": np.array([r["label"] == r["gold"] for r in rows], dtype=bool),
        "refused": np.array([r["refused"] for r in rows], dtype=bool),
        "latency": np.array([r["latency_s"] for r in rows], dtype=float),
        "cost": np.array([r["cost_usd"] for r in rows], dtype=float),
    }


def summarize(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    dev = _arrays([r for r in rows if r["split"] == "dev"])
    test_rows = [r for r in rows if r["split"] == "test"]
    test = _arrays(test_rows)
    allr = _arrays(rows)
    threshold = metrics.choose_threshold(dev["confidence"], dev["correct"], dev["refused"], target)
    cov, acc = (
        metrics.at_threshold(threshold, test["confidence"], test["correct"], test["refused"])
        if threshold is not None
        else (None, None)
    )
    p50, p95 = metrics.latency_percentiles(allr["latency"])
    return {
        "model": rows[0]["model"] if rows else None,
        "method": rows[0]["method"] if rows else None,
        "n": len(rows),
        "n_dev": int(dev["confidence"].size),
        "n_test": len(test_rows),
        "accuracy": metrics.accuracy(test["correct"]),
        "ece": metrics.ece(test["confidence"], test["correct"]),
        "aurc": metrics.aurc(test["confidence"], test["correct"]),
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "cost_per_1k_usd": metrics.cost_per_1k(allr["cost"]),
        "refusals": int(allr["refused"].sum()),
        "target_accuracy": target,
        "dev_threshold": threshold,
        "test_coverage_at_threshold": cov,
        "test_accuracy_at_threshold": acc,
        "spend_usd": float(allr["cost"].sum()),
    }


def temperature_scaled(rows: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    """Fit T on answered dev rows; return T and rows whose confidence is the top temperature-scaled probability."""
    answered = [i for i, r in enumerate(rows) if not r["refused"]]
    probs = np.array([[rows[i]["probs"][q] for q in QUEUE_NAMES] for i in answered], dtype=float)
    probs = probs.reshape(-1, len(QUEUE_NAMES))
    gold = np.array([QUEUE_NAMES.index(rows[i]["gold"]) for i in answered], dtype=int)
    dev = np.array([rows[i]["split"] == "dev" for i in answered], dtype=bool)
    t = calibrate.fit_temperature(probs[dev], gold[dev])
    scaled = [dict(r) for r in rows]
    for i, conf in zip(answered, calibrate.scale(probs, t).max(axis=1), strict=True):
        scaled[i]["confidence"] = float(conf)
    return t, scaled


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reliability(methods: dict[str, list[dict[str, Any]]], path: Path) -> None:
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, facecolor=SURFACE)
    for i, (ax, (name, rows)) in enumerate(zip(axes[0], methods.items(), strict=True)):
        test = _arrays([r for r in rows if r["split"] == "test"])
        bins = metrics.reliability_bins(test["confidence"], test["correct"])
        filled = [b for b in bins if b.count]
        _style(ax)
        ax.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=1, linestyle="--", label="perfect calibration")
        ax.bar(
            [b.lo for b in filled], [b.accuracy for b in filled], width=1 / len(bins), align="edge",
            color=SERIES[i % len(SERIES)], edgecolor=SURFACE, linewidth=2, label="accuracy in bin",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence", color=TEXT_MUTED, fontsize=9)
        if i == 0:
            ax.set_ylabel("accuracy", color=TEXT_MUTED, fontsize=9)
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_MUTED, loc="upper left")
        ece = metrics.ece(test["confidence"], test["correct"])
        title = name.replace(CALIBRATED_SUFFIX, "\n" + CALIBRATED_SUFFIX.strip())
        ax.set_title(f"{title}\nECE {ece:.3f} (test, n={test['confidence'].size})", color=TEXT, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_coverage(methods: dict[str, list[dict[str, Any]]], summary: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    _style(ax)
    for i, (name, rows) in enumerate(methods.items()):
        color = SERIES[i % len(SERIES)]
        test = _arrays([r for r in rows if r["split"] == "test"])
        _, cov, acc = metrics.coverage_curve(test["confidence"], test["correct"], test["refused"])
        ax.plot(cov, acc, color=color, linewidth=2, label=name, drawstyle="steps-post")
        s = summary[name]
        if s["dev_threshold"] is not None:
            ax.plot(
                s["test_coverage_at_threshold"], s["test_accuracy_at_threshold"], "o", markersize=8,
                color=color, markeredgecolor=SURFACE, markeredgewidth=2,
            )
            ax.annotate(
                f"t={s['dev_threshold']:.3f}", (s["test_coverage_at_threshold"], s["test_accuracy_at_threshold"]),
                textcoords="offset points", xytext=(6, -12), fontsize=8, color=TEXT_MUTED,
            )
    target = next(iter(summary.values()))["target_accuracy"] if summary else None
    if target is not None:
        ax.axhline(target, color=TEXT_MUTED, linewidth=1, linestyle=":", label=f"target {target:.2f}")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("coverage (share auto-routed)", color=TEXT_MUTED, fontsize=9)
    ax.set_ylabel("accuracy on auto-routed", color=TEXT_MUTED, fontsize=9)
    ax.set_title("Auto-route vs human review (test split; dots = dev-chosen threshold)", color=TEXT, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_MUTED, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_report(raw_files: list[Path], summary_path: Path, docs_dir: Path, target: float = 0.98) -> dict[str, Any]:
    methods = {p.stem: load_jsonl(p) for p in sorted(raw_files)}
    per_method = {name: summarize(rows, target) for name, rows in methods.items()}
    series, series_summary = dict(methods), dict(per_method)
    for name, rows in methods.items():
        if per_method[name]["method"] != "decisions":
            continue
        t, scaled = temperature_scaled(rows)
        cal = summarize(scaled, target)
        per_method[name]["temperature"] = t
        per_method[name].update({f"{k}_calibrated": cal[k] for k in CALIBRATED_KEYS})
        series[name + CALIBRATED_SUFFIX] = scaled
        series_summary[name + CALIBRATED_SUFFIX] = cal
    summary = {
        "target_accuracy": target,
        "total_spend_usd": float(sum(s["spend_usd"] for s in per_method.values())),
        "methods": per_method,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    docs_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(series, docs_dir / "reliability.png")
    plot_coverage(series, series_summary, docs_dir / "coverage.png")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize raw eval runs into metrics and charts.")
    p.add_argument("raw", type=Path, nargs="*", help="raw jsonl files (default: results/raw/*.jsonl)")
    p.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    p.add_argument("--docs", type=Path, default=Path("docs"))
    p.add_argument("--target", type=float, default=0.98)
    args = p.parse_args(argv)
    raw = args.raw or sorted(Path("results/raw").glob("*.jsonl"))
    if not raw:
        p.error("no raw files found")
    summary = build_report(raw, args.summary, args.docs, args.target)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import random

from decision_triage.report import build_report, main
from decision_triage.taxonomy import QUEUE_NAMES


def fake_rows(method: str, seed: int, n: int = 60) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        gold = QUEUE_NAMES[i % len(QUEUE_NAMES)]
        conf = rng.uniform(0.4, 1.0)
        refused = i == 5
        label = None if refused else (gold if rng.random() < conf else "other")
        probs = dict.fromkeys(QUEUE_NAMES, 0.0 if refused else (1 - conf) / (len(QUEUE_NAMES) - 1))
        if label is not None:
            probs[label] = conf
        rows.append({
            "method": method, "model": "m", "id": f"x{i}", "gold": gold, "split": "dev" if i % 3 == 0 else "test",
            "label": label, "probs": probs, "confidence": 0.0 if refused else conf, "refused": refused,
            "latency_s": 0.1 + i / 1000, "cost_usd": 1e-5,
        })
    return rows


def underconfident_decisions_rows(softmax_data, n: int = 900) -> list[dict]:
    probs, gold = softmax_data(n, 0.5, seed=7)
    rows = []
    for i, (p, g) in enumerate(zip(probs, gold, strict=True)):
        refused = i == 4
        label = None if refused else QUEUE_NAMES[int(p.argmax())]
        rows.append({
            "method": "decisions", "model": "m", "id": f"d{i}", "gold": QUEUE_NAMES[int(g)],
            "split": "dev" if i % 3 == 0 else "test", "label": label,
            "probs": dict.fromkeys(QUEUE_NAMES, 0.0) if refused else dict(zip(QUEUE_NAMES, p.tolist(), strict=True)),
            "confidence": 0.0 if refused else float(p.max()), "refused": refused,
            "latency_s": 0.1, "cost_usd": 1e-5,
        })
    return rows


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "decisions.jsonl", fake_rows("decisions", 1))
    write(raw / "responses-nano.jsonl", fake_rows("responses", 2))
    summary_path = tmp_path / "results" / "summary.json"
    docs = tmp_path / "docs"

    rc = main([str(p) for p in raw.glob("*.jsonl")] + ["--summary", str(summary_path), "--docs", str(docs)])
    assert rc == 0

    summary = json.loads(summary_path.read_text())
    assert set(summary["methods"]) == {"decisions", "responses-nano"}
    s = summary["methods"]["decisions"]
    for key in (
        "n", "accuracy", "ece", "latency_p50_s", "latency_p95_s", "cost_per_1k_usd", "refusals",
        "dev_threshold", "test_coverage_at_threshold", "test_accuracy_at_threshold", "spend_usd", "aurc",
        "temperature", "ece_calibrated", "aurc_calibrated", "dev_threshold_calibrated",
        "test_coverage_at_threshold_calibrated", "test_accuracy_at_threshold_calibrated",
    ):
        assert key in s
    assert "aurc" in summary["methods"]["responses-nano"]
    assert "temperature" not in summary["methods"]["responses-nano"]
    assert s["n"] == 60 and s["n_dev"] == 20 and s["n_test"] == 40
    assert s["refusals"] == 1
    assert abs(s["cost_per_1k_usd"] - 0.01) < 1e-9
    assert abs(summary["total_spend_usd"] - 2 * 60 * 1e-5) < 1e-9
    for name in ("reliability.png", "coverage.png"):
        data = (docs / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000


def test_threshold_none_when_target_unreachable(tmp_path):
    rows = fake_rows("decisions", 3)
    for r in rows:
        r["label"] = "wrong"
    write(tmp_path / "d.jsonl", rows)
    summary = build_report([tmp_path / "d.jsonl"], tmp_path / "s.json", tmp_path / "docs")
    s = summary["methods"]["d"]
    assert s["dev_threshold"] is None and s["test_coverage_at_threshold"] is None
    assert (tmp_path / "docs" / "coverage.png").exists()


def png_width(path) -> int:
    return int.from_bytes(path.read_bytes()[16:20], "big")


def test_report_temperature_scales_decisions_only(tmp_path, softmax_data):
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "decisions-m.jsonl", underconfident_decisions_rows(softmax_data))
    write(raw / "responses-a.jsonl", fake_rows("responses", 2))
    write(raw / "responses-b.jsonl", fake_rows("responses", 3))
    docs = tmp_path / "docs"
    summary = build_report(sorted(raw.glob("*.jsonl")), tmp_path / "s.json", docs)

    d = summary["methods"]["decisions-m"]
    assert d["temperature"] < 1.0
    assert d["ece"] > 0.05 and d["ece_calibrated"] < d["ece"] / 2
    assert d["dev_threshold_calibrated"] is not None
    assert 0 < d["test_coverage_at_threshold_calibrated"] <= 1
    assert d["test_accuracy_at_threshold_calibrated"] >= 0.9
    assert 0 <= d["aurc_calibrated"] < 1 and 0 <= d["aurc"] < 1
    for name in ("responses-a", "responses-b"):
        assert not any(k.endswith("_calibrated") or k == "temperature" for k in summary["methods"][name])
    assert json.loads((tmp_path / "s.json").read_text()) == summary
    assert png_width(docs / "reliability.png") == 4 * 4 * 150

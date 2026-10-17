import copy
import json
from pathlib import Path

import pytest
from render_readme import TOKEN, RenderError, build_context, main, render
from test_report import fake_rows, underconfident_decisions_rows, write

from decision_triage.report import build_report

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "README.template.md"


def method_summary(p50: float, cost: float, **extra: float) -> dict:
    return {
        "n": 700, "n_dev": 210, "n_test": 490, "accuracy": 0.9347, "ece": 0.1666, "aurc": 0.012,
        "latency_p50_s": p50, "latency_p95_s": p50 * 1.4, "cost_per_1k_usd": cost, "dev_threshold": 0.631,
        "test_coverage_at_threshold": 0.7306, "test_accuracy_at_threshold": 0.986, **extra,
    }


@pytest.fixture
def summary() -> dict:
    return {
        "target_accuracy": 0.98,
        "total_spend_usd": 0.08348,
        "methods": {
            "decisions-gpt-6-luna": method_summary(
                0.1305, 0.02308, temperature=0.42, ece_calibrated=0.03, aurc_calibrated=0.011,
                dev_threshold_calibrated=0.9, test_coverage_at_threshold_calibrated=0.75,
                test_accuracy_at_threshold_calibrated=0.985,
            ),
            "responses-gpt-6-luna": method_summary(1.0646, 0.03046),
            "responses-gpt-5.4-nano": method_summary(0.7901, 0.06572),
        },
    }


def test_every_template_token_is_replaced(summary):
    template = TEMPLATE.read_text()
    tokens = set(TOKEN.findall(template))
    assert {"decisions.accuracy", "decisions.ece", "decisions.ece_calibrated", "decisions.temperature"} <= tokens
    out = render(template, build_context(summary))
    assert "{{" not in out and "}}" not in out
    assert "**Decisions: 8.2x lower p50 latency (130 ms vs 1065 ms) and 24% lower cost per 1k" in out
    row = "| decisions gpt-6-luna (temperature-scaled) | 93.5% | 0.030 | 0.0110 | 130 / 183 | $0.0231 | 75.0%, 98.5% |"
    assert row in out
    assert "T=0.42" in out and "Total computed spend for all runs: $0.0835" in out


def test_unknown_token_fails_loudly(summary):
    with pytest.raises(RenderError, match="unknown README tokens: decisions.f1"):
        render("x {{decisions.f1}} {{decisions.accuracy}}", build_context(summary))


def test_malformed_token_fails_loudly(summary):
    with pytest.raises(RenderError, match="malformed"):
        render("x {{ Decisions.Accuracy }}", build_context(summary))


def test_missing_summary_key_fails_loudly(summary):
    del summary["methods"]["decisions-gpt-6-luna"]["temperature"]
    with pytest.raises(RenderError, match="missing 'temperature'"):
        build_context(summary)


@pytest.mark.parametrize(
    ("key", "value", "claim"),
    [
        ("latency_p50_s", 2.0, "p50 latency"),
        ("cost_per_1k_usd", 1.0, "cost per 1k"),
        ("temperature", 1.5, "under-confident"),
        ("ece_calibrated", 0.5, "lowers Decisions ECE"),
    ],
)
def test_broken_claim_fails(summary, key, value, claim):
    broken = copy.deepcopy(summary)
    broken["methods"]["decisions-gpt-6-luna"][key] = value
    with pytest.raises(RenderError, match=claim):
        build_context(broken)


def test_report_then_render_end_to_end(tmp_path, softmax_data):
    raw = tmp_path / "raw"
    raw.mkdir()
    decisions = underconfident_decisions_rows(softmax_data)
    for r in decisions:
        r["latency_s"], r["cost_usd"] = 0.1, 2e-5
    responses = fake_rows("responses", 2)
    for r in responses:
        r["latency_s"], r["cost_usd"] = 1.0, 3e-5
    write(raw / "decisions-gpt-6-luna.jsonl", decisions)
    write(raw / "responses-gpt-6-luna.jsonl", responses)
    write(raw / "responses-gpt-5.4-nano.jsonl", fake_rows("responses", 3))
    summary_path = tmp_path / "summary.json"
    build_report(sorted(raw.glob("*.jsonl")), summary_path, tmp_path / "docs")

    out = tmp_path / "README.md"
    args = ["--template", str(TEMPLATE), "--summary", str(summary_path), "--out", str(out)]
    assert main([*args, "--check"]) == 1
    assert main(args) == 0
    text = out.read_text()
    assert "{{" not in text and "**Decisions: 10.0x lower p50 latency (100 ms vs 1000 ms) and 33% lower" in text
    assert main([*args, "--check"]) == 0
    out.write_text(text.replace("10.0x", "11.0x"))
    assert main([*args, "--check"]) == 1

    s = json.loads(summary_path.read_text())
    s["methods"]["decisions-gpt-6-luna"].pop("ece_calibrated")
    summary_path.write_text(json.dumps(s))
    assert main(args) == 2

import json
import threading

import pytest

from decision_triage import run_eval
from decision_triage.prediction import Prediction
from decision_triage.run_eval import load_jsonl, run


class Transient(Exception):
    pass


class FakeClassifier:
    method = "fake"
    model = "fake-1"

    def __init__(self, cost: float = 0.01, estimate: float = 0.01, fail_first: int = 0) -> None:
        self.cost = cost
        self.estimate = estimate
        self.fail_first = fail_first
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def classify(self, item_id: str, text: str) -> Prediction:
        with self._lock:
            self.calls.append(item_id)
            if self.fail_first > 0:
                self.fail_first -= 1
                raise Transient("boom")
        return Prediction(
            id=item_id, label="spam", probs={"spam": 1.0}, confidence=1.0, api_confidence=None, latency_s=0.01,
            input_tokens=10, output_tokens=0, cost_usd=self.cost, refused=False,
        )

    def estimate_cost(self, text: str) -> float:
        return self.estimate


def items(n: int) -> list[dict]:
    return [{"id": f"m{i}", "text": f"t{i}", "label": "spam", "split": "test"} for i in range(n)]


def quiet(_: str) -> None:
    pass


@pytest.mark.parametrize("concurrency", [1, 4])
def test_budget_guard_stops_before_overspending(tmp_path, concurrency):
    out = tmp_path / "r.jsonl"
    clf = FakeClassifier(cost=0.01, estimate=0.01)
    stats = run(clf, items(10), out, concurrency=concurrency, budget_usd=0.035, transient=(Transient,), log=quiet)
    assert stats.budget_stopped
    assert stats.done == 3
    assert len(clf.calls) == 3
    assert stats.run_spend_usd == pytest.approx(0.03)
    assert len(load_jsonl(out)) == 3


def test_budget_guard_learns_from_actual_costs(tmp_path):
    clf = FakeClassifier(cost=0.02, estimate=0.001)
    stats = run(clf, items(10), tmp_path / "r.jsonl", concurrency=1, budget_usd=0.05, transient=(Transient,), log=quiet)
    assert stats.done == 2
    assert stats.run_spend_usd <= 0.05


def test_budget_includes_prior_spend(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text(json.dumps({"id": "m0", "cost_usd": 0.03}) + "\n")
    clf = FakeClassifier(cost=0.01, estimate=0.01)
    stats = run(clf, items(10), out, concurrency=1, budget_usd=0.05, transient=(Transient,), log=quiet)
    assert stats.prior_spend_usd == pytest.approx(0.03)
    assert stats.done == 2


def test_resume_skips_done_ids(tmp_path):
    out = tmp_path / "r.jsonl"
    run(FakeClassifier(), items(2), out, budget_usd=1.0, transient=(Transient,), log=quiet)
    clf = FakeClassifier()
    stats = run(clf, items(5), out, concurrency=2, budget_usd=1.0, transient=(Transient,), log=quiet)
    assert sorted(clf.calls) == ["m2", "m3", "m4"]
    assert stats.skipped == 2
    rows = load_jsonl(out)
    assert sorted(r["id"] for r in rows) == [f"m{i}" for i in range(5)]
    assert rows[0]["gold"] == "spam" and rows[0]["split"] == "test" and rows[0]["method"] == "fake"


def test_retries_transient_errors(tmp_path):
    sleeps: list[float] = []
    clf = FakeClassifier(fail_first=2)
    stats = run(
        clf, items(1), tmp_path / "r.jsonl", budget_usd=1.0, transient=(Transient,), sleep=sleeps.append, log=quiet
    )
    assert stats.done == 1 and stats.failed == 0
    assert len(clf.calls) == 3 and len(sleeps) == 2


def test_gives_up_after_max_retries_without_writing(tmp_path):
    out = tmp_path / "r.jsonl"
    clf = FakeClassifier(fail_first=4)
    stats = run(clf, items(1), out, budget_usd=1.0, transient=(Transient,), sleep=lambda s: None, log=quiet)
    assert stats.failed == 1 and stats.done == 0
    assert len(clf.calls) == 4
    assert load_jsonl(out) == []


def test_cli_main_with_limit(tmp_path, monkeypatch, capsys):
    data = tmp_path / "eval.jsonl"
    data.write_text("".join(json.dumps(it) + "\n" for it in items(5)))
    out = tmp_path / "raw" / "fake.jsonl"
    monkeypatch.setattr(run_eval, "build_classifier", lambda args: FakeClassifier())
    rc = run_eval.main(["--method", "decisions", "--data", str(data), "--out", str(out), "--limit", "3"])
    assert rc == 0
    assert sorted(r["id"] for r in load_jsonl(out)) == ["m0", "m1", "m2"]
    assert "spend:" in capsys.readouterr().err


def test_build_classifier_offline(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    parse = run_eval.argparse.Namespace
    d = run_eval.build_classifier(parse(method="decisions", model=None, reasoning_effort="none"))
    assert (d.method, d.model) == ("decisions", "gpt-6-luna")
    r = run_eval.build_classifier(parse(method="responses", model="gpt-6-luna", reasoning_effort="omit"))
    assert (r.method, r.model, r.reasoning_effort) == ("responses", "gpt-6-luna", None)
    assert d.client.max_retries == 0


def test_non_transient_error_not_retried(tmp_path):
    class Broken(FakeClassifier):
        def classify(self, item_id, text):
            self.calls.append(item_id)
            raise ValueError("bad request")

    clf = Broken()
    stats = run(clf, items(2), tmp_path / "r.jsonl", budget_usd=1.0, transient=(Transient,), log=quiet)
    assert stats.failed == 2 and clf.calls == ["m0", "m1"]

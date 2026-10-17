import pytest
from conftest import FakeOpenAI, make_decision

from decision_triage.decisions_client import DecisionsClassifier
from decision_triage.taxonomy import QUEUE_NAMES


def test_request_shape():
    client = FakeOpenAI(make_decision({"billing": 0.9, "orders": 0.1}))
    DecisionsClassifier(client).classify("m1", "Where is my invoice?")
    (call,) = client.decisions.calls
    assert call["model"] == "gpt-6-luna"
    assert call["input"] == "Where is my invoice?"
    (q,) = call["questions"]
    assert q["type"] == "choice"
    assert q["name"] == "queue"
    assert q["instructions"]
    assert [c["value"] for c in q["choices"]] == list(QUEUE_NAMES)
    assert len(QUEUE_NAMES) == 7
    assert all(c["description"] for c in q["choices"])


def test_parse_choice():
    client = FakeOpenAI(make_decision({"billing": 0.7, "orders": 0.2, "spam": 0.1}, input_tokens=1_000_000))
    pred = DecisionsClassifier(client).classify("m1", "x")
    assert pred.label == "billing"
    assert not pred.refused
    assert set(pred.probs) == set(QUEUE_NAMES)
    assert pred.probs["billing"] == 0.7
    assert pred.probs["account"] == 0.0
    assert pred.confidence == 0.7
    assert pred.api_confidence == 0.5
    assert pred.input_tokens == 1_000_000
    assert pred.output_tokens == 0
    assert pred.cost_usd == pytest.approx(0.10)
    assert pred.latency_s >= 0
    assert pred.raw["answers"][0]["choice"] == "billing"


def test_parse_refusal():
    pred = DecisionsClassifier(FakeOpenAI(make_decision(None))).classify("m2", "x")
    assert pred.refused
    assert pred.label is None
    assert pred.confidence == 0.0
    assert pred.api_confidence is None
    assert pred.cost_usd > 0


def test_estimate_is_positive_and_small():
    est = DecisionsClassifier(FakeOpenAI(None)).estimate_cost("hello " * 50)
    assert 0 < est < 0.001

import json

import pytest
from conftest import FakeOpenAI, make_response

from decision_triage.responses_client import ResponsesClassifier
from decision_triage.taxonomy import QUEUE_NAMES


def test_request_shape():
    client = FakeOpenAI(make_response('{"queue":"spam","confidence":0.9}', 100, 0, 10, 0))
    ResponsesClassifier(client).classify("m1", "WIN a prize")
    (call,) = client.responses.calls
    assert call["model"] == "gpt-5.4-nano"
    assert call["input"] == "WIN a prize"
    assert call["reasoning"] == {"effort": "none"}
    fmt = call["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"]["properties"]["queue"]["enum"] == list(QUEUE_NAMES)
    assert fmt["schema"]["required"] == ["queue", "confidence"]
    for name in QUEUE_NAMES:
        assert name in call["instructions"]


def test_reasoning_omitted_and_configurable():
    client = FakeOpenAI(make_response('{"queue":"spam","confidence":0.9}', 1, 0, 1, 0))
    assert "reasoning" not in ResponsesClassifier(client, reasoning_effort=None).request("x")
    assert ResponsesClassifier(client, reasoning_effort="low").request("x")["reasoning"] == {"effort": "low"}


def test_parse_and_cost_with_cached_and_reasoning_tokens():
    out = json.dumps({"queue": "orders", "confidence": 0.82})
    resp = make_response(out, input_tokens=1_000_000, cached=400_000, output_tokens=200_000, reasoning=150_000)
    pred = ResponsesClassifier(FakeOpenAI(resp)).classify("m1", "x")
    assert pred.label == "orders"
    assert pred.confidence == 0.82
    assert pred.probs["orders"] == 0.82
    assert sum(pred.probs.values()) == pytest.approx(1.0)
    assert pred.probs["spam"] == pytest.approx(0.18 / 6)
    assert pred.reasoning_tokens == 150_000
    assert pred.cached_tokens == 400_000
    assert pred.output_tokens == 200_000
    expected = 0.6 * 0.20 + 0.4 * 0.02 + 0.2 * 1.25
    assert pred.cost_usd == pytest.approx(expected)


def test_luna_pricing():
    resp = make_response('{"queue":"other","confidence":1}', 1_000_000, 0, 1_000_000, 0)
    pred = ResponsesClassifier(FakeOpenAI(resp), model="gpt-6-luna").classify("m1", "x")
    assert pred.cost_usd == pytest.approx(0.10 + 0.50)


@pytest.mark.parametrize("text", ["", "not json", '{"queue":"nope","confidence":0.5}', '{"queue":"spam"}'])
def test_unparseable_output_is_refusal(text):
    pred = ResponsesClassifier(FakeOpenAI(make_response(text, 10, 0, 5, 0))).classify("m1", "x")
    assert pred.refused and pred.label is None and pred.confidence == 0.0
    assert pred.cost_usd > 0


def test_confidence_clipped():
    pred = ResponsesClassifier(FakeOpenAI(make_response('{"queue":"spam","confidence":1.4}', 1, 0, 1, 0))).classify(
        "m1", "x"
    )
    assert pred.confidence == 1.0

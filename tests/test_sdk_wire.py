import json

import httpx2
import pytest
from openai import OpenAI

from decision_triage.decisions_client import DecisionsClassifier
from decision_triage.responses_client import ResponsesClassifier
from decision_triage.taxonomy import QUEUE_NAMES

USAGE = {
    "input_tokens": 200,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
    "output_tokens": 12,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 212,
}


def offline_client(body: dict, sent: list[httpx2.Request]) -> OpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=body)

    return OpenAI(api_key="sk-test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_decisions_over_sdk_wire():
    body = {
        "model": "gpt-6-luna",
        "answers": [{
            "type": "choice", "name": "queue", "choice": "orders", "confidence": 0.93,
            "probabilities": [{"value": "orders", "probability": 0.96}, {"value": "billing", "probability": 0.04}],
        }],
        "usage": {**USAGE, "output_tokens": 0},
    }
    sent: list[httpx2.Request] = []
    pred = DecisionsClassifier(offline_client(body, sent)).classify("m1", "Where is my parcel?")
    (req,) = sent
    assert req.method == "POST" and req.url.path.endswith("/decisions")
    payload = json.loads(req.content)
    assert payload["model"] == "gpt-6-luna" and payload["input"] == "Where is my parcel?"
    assert [c["value"] for c in payload["questions"][0]["choices"]] == list(QUEUE_NAMES)
    assert pred.label == "orders" and pred.confidence == 0.96 and pred.api_confidence == 0.93
    assert pred.cost_usd == pytest.approx(200 * 0.10 / 1e6)


def test_responses_over_sdk_wire():
    body = {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-5.4-nano", "status": "completed",
        "output": [{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": '{"queue":"spam","confidence":0.7}', "annotations": []}],
        }],
        "usage": USAGE,
    }
    sent: list[httpx2.Request] = []
    pred = ResponsesClassifier(offline_client(body, sent)).classify("m1", "Free prize!!")
    (req,) = sent
    assert req.url.path.endswith("/responses")
    payload = json.loads(req.content)
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["store"] is False
    assert pred.label == "spam" and pred.confidence == 0.7
    assert pred.cost_usd == pytest.approx((200 * 0.20 + 12 * 1.25) / 1e6)

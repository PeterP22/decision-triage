import json
import time
from typing import Any

from decision_triage import pricing
from decision_triage.prediction import Prediction, estimate_tokens
from decision_triage.taxonomy import QUEUE_NAMES, TASK, queue_list_text

INSTRUCTIONS = (
    f"{TASK}\n\nQueues:\n{queue_list_text()}\n\n"
    "Return the queue and your confidence, from 0 to 1, that the queue is correct."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queue": {"type": "string", "enum": list(QUEUE_NAMES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["queue", "confidence"],
    "additionalProperties": False,
}

TEXT_FORMAT: dict[str, Any] = {
    "format": {"type": "json_schema", "name": "triage", "strict": True, "schema": SCHEMA}
}

ESTIMATED_OUTPUT_TOKENS = 200


def spread_probs(label: str, confidence: float) -> dict[str, float]:
    rest = (1.0 - confidence) / (len(QUEUE_NAMES) - 1)
    return {q: confidence if q == label else rest for q in QUEUE_NAMES}


def parse_response(item_id: str, response: Any, latency_s: float, price: pricing.Price) -> Prediction:
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cached = usage.input_tokens_details.cached_tokens or 0
    reasoning = usage.output_tokens_details.reasoning_tokens or 0
    cost = pricing.cost_usd(price, input_tokens, cached, output_tokens)
    text = getattr(response, "output_text", "") or ""
    raw = {"output_text": text, "usage": usage.model_dump(mode="json") if hasattr(usage, "model_dump") else {}}
    common = dict(
        id=item_id, latency_s=latency_s, input_tokens=input_tokens, output_tokens=output_tokens,
        cost_usd=cost, cached_tokens=cached, reasoning_tokens=reasoning, raw=raw, api_confidence=None,
    )
    try:
        data = json.loads(text)
        label = data["queue"]
        confidence = min(max(float(data["confidence"]), 0.0), 1.0)
        if label not in QUEUE_NAMES:
            raise ValueError(label)
    except (ValueError, KeyError, TypeError):
        return Prediction(label=None, probs=dict.fromkeys(QUEUE_NAMES, 0.0), confidence=0.0, refused=True, **common)
    return Prediction(
        label=label, probs=spread_probs(label, confidence), confidence=confidence, refused=False, **common
    )


class ResponsesClassifier:
    method = "responses"

    def __init__(self, client: Any, model: str = "gpt-5.4-nano", reasoning_effort: str | None = "none") -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.price = pricing.RESPONSES[model]
        self._overhead_tokens = estimate_tokens(INSTRUCTIONS + json.dumps(SCHEMA))

    def request(self, text: str) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.model,
            "instructions": INSTRUCTIONS,
            "input": text,
            "text": TEXT_FORMAT,
            "store": False,
        }
        if self.reasoning_effort is not None:
            req["reasoning"] = {"effort": self.reasoning_effort}
        return req

    def classify(self, item_id: str, text: str) -> Prediction:
        start = time.perf_counter()
        response = self.client.responses.create(**self.request(text))
        return parse_response(item_id, response, time.perf_counter() - start, self.price)

    def estimate_cost(self, text: str) -> float:
        tokens = estimate_tokens(text) + self._overhead_tokens
        return 2 * pricing.cost_usd(self.price, tokens, 0, ESTIMATED_OUTPUT_TOKENS)

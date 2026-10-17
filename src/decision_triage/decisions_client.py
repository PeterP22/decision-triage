import json
import time
from typing import Any

from decision_triage import pricing
from decision_triage.prediction import Prediction, estimate_tokens
from decision_triage.taxonomy import QUEUE_NAMES, QUEUES, TASK

QUESTION_NAME = "queue"


def build_question() -> dict[str, Any]:
    return {
        "type": "choice",
        "name": QUESTION_NAME,
        "instructions": TASK,
        "choices": [{"value": name, "description": desc} for name, desc in QUEUES.items()],
    }


def parse_decision(item_id: str, response: Any, latency_s: float, price: pricing.Price) -> Prediction:
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens or 0
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    cost = pricing.cost_usd(price, input_tokens, cached, output_tokens)
    raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}

    answer = response.answers[0]
    probs = dict.fromkeys(QUEUE_NAMES, 0.0)
    if answer.type != "choice":
        return Prediction(
            id=item_id, label=None, probs=probs, confidence=0.0, api_confidence=None,
            latency_s=latency_s, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, refused=True, cached_tokens=cached, raw=raw,
        )
    for p in answer.probabilities:
        if p.value in probs:
            probs[str(p.value)] = float(p.probability)
    return Prediction(
        id=item_id,
        label=str(answer.choice),
        probs=probs,
        confidence=max(probs.values()),
        api_confidence=float(answer.confidence),
        latency_s=latency_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        refused=False,
        cached_tokens=cached,
        raw=raw,
    )


class DecisionsClassifier:
    method = "decisions"

    def __init__(self, client: Any, model: str = "gpt-6-luna") -> None:
        self.client = client
        self.model = model
        self.price = pricing.DECISIONS[model]
        self.question = build_question()
        self._overhead_tokens = estimate_tokens(json.dumps(self.question))

    def request(self, text: str) -> dict[str, Any]:
        return {"model": self.model, "input": text, "questions": [self.question]}

    def classify(self, item_id: str, text: str) -> Prediction:
        start = time.perf_counter()
        response = self.client.decisions.create(**self.request(text))
        return parse_decision(item_id, response, time.perf_counter() - start, self.price)

    def estimate_cost(self, text: str) -> float:
        return 2 * pricing.cost_usd(self.price, estimate_tokens(text) + self._overhead_tokens)

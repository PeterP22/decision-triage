from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class Prediction:
    id: str
    label: str | None
    probs: dict[str, float]
    confidence: float
    api_confidence: float | None
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    refused: bool
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Classifier(Protocol):
    method: str
    model: str

    def classify(self, item_id: str, text: str) -> Prediction: ...

    def estimate_cost(self, text: str) -> float: ...


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1

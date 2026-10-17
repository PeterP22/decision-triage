from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from openai.types import Decision
from openai.types.responses import ResponseUsage


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture
def softmax_data() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Reported probs softmax(z) with gold drawn from softmax(z / true_t): temperature scaling should recover true_t."""

    def make(n: int, true_t: float, k: int = 7, scale: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, scale, size=(n, k))
        truth = _softmax(z / true_t)
        gold = (truth.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
        return _softmax(z), gold

    return make


def make_decision(probs: dict[str, float] | None, input_tokens: int = 120, cached: int = 0) -> Decision:
    usage = {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": cached},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens,
    }
    if probs is None:
        answer: dict[str, Any] = {"type": "refusal", "name": "queue"}
    else:
        top = max(probs, key=probs.__getitem__)
        answer = {
            "type": "choice",
            "name": "queue",
            "choice": top,
            "confidence": 0.5,
            "probabilities": [{"value": k, "probability": v} for k, v in probs.items()],
        }
    return Decision.model_validate({"answers": [answer], "model": "gpt-6-luna", "usage": usage})


def make_response(output_text: str, input_tokens: int, cached: int, output_tokens: int, reasoning: int) -> Any:
    usage = ResponseUsage.model_validate({
        "input_tokens": input_tokens,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output_tokens,
    })
    return SimpleNamespace(output_text=output_text, usage=usage)


class Recorder:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


class FakeOpenAI:
    def __init__(self, result: Any) -> None:
        self.decisions = Recorder(result)
        self.responses = Recorder(result)

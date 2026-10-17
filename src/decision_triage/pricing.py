"""USD prices per 1M tokens.

Responses API: https://developers.openai.com/api/docs/pricing (read 2026-10-17).
Decisions API: https://developers.openai.com/api/docs/guides/decisions (read 2026-10-17);
no output charge and no separate cached-input rate is published, so all input
tokens are billed at the full input rate here (conservative).
Reasoning tokens are included in output_tokens and bill as output.
"""

from dataclasses import dataclass

PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
DECISIONS_SOURCE = "https://developers.openai.com/api/docs/guides/decisions"
PRICING_DATE = "2026-10-17"


@dataclass(frozen=True)
class Price:
    input: float
    cached_input: float
    output: float


RESPONSES: dict[str, Price] = {
    "gpt-5.4-nano": Price(input=0.20, cached_input=0.02, output=1.25),
    "gpt-6-luna": Price(input=0.10, cached_input=0.01, output=0.50),
}

DECISIONS: dict[str, Price] = {
    "gpt-6-luna": Price(input=0.10, cached_input=0.10, output=0.0),
}


def cost_usd(price: Price, input_tokens: int, cached_tokens: int = 0, output_tokens: int = 0) -> float:
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * price.input + cached_tokens * price.cached_input + output_tokens * price.output) / 1e6

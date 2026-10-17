QUEUES: dict[str, str] = {
    "billing": "Invoices, payments, refunds, cancellation fees.",
    "orders": "Placing, changing, cancelling or tracking orders; delivery; shipping address.",
    "account": "Create, edit, delete or switch account; password; registration; newsletter subscription.",
    "feedback": "Complaints and reviews.",
    "human_agent": "Asks to reach customer service or a human.",
    "spam": "Unsolicited promotions, prize or scam messages.",
    "other": "Personal or off-topic messages that are not a customer request.",
}

QUEUE_NAMES: tuple[str, ...] = tuple(QUEUES)

TASK = "Route this inbound inbox message to the single queue that should handle it."


def queue_list_text() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in QUEUES.items())

import argparse
import json
import random
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_triage.prediction import Classifier, Prediction

MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 5


def default_transient() -> tuple[type[BaseException], ...]:
    import openai

    return (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class Budget:
    def __init__(self, limit_usd: float, spent_usd: float = 0.0) -> None:
        self.limit = limit_usd
        self.spent = spent_usd
        self.reserved = 0.0
        self._lock = threading.Lock()

    def try_reserve(self, estimate: float) -> bool:
        with self._lock:
            if self.spent + self.reserved + estimate > self.limit:
                return False
            self.reserved += estimate
            return True

    def settle(self, estimate: float, actual: float) -> None:
        with self._lock:
            self.reserved -= estimate
            self.spent += actual


@dataclass
class RunStats:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    prior_spend_usd: float = 0.0
    run_spend_usd: float = 0.0
    budget_stopped: bool = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def call_with_retries(
    fn: Callable[[], Prediction],
    transient: tuple[type[BaseException], ...],
    sleep: Callable[[float], None] = time.sleep,
    retries: int = MAX_RETRIES,
) -> Prediction:
    for attempt in range(retries + 1):
        try:
            return fn()
        except transient:
            if attempt == retries:
                raise
            sleep(2**attempt + random.random())
    raise AssertionError("unreachable")


def run(
    classifier: Classifier,
    items: Iterable[dict[str, Any]],
    out_path: Path,
    *,
    concurrency: int = 4,
    budget_usd: float = 1.0,
    transient: tuple[type[BaseException], ...] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> RunStats:
    transient = default_transient() if transient is None else transient
    existing = load_jsonl(out_path)
    done_ids = {r["id"] for r in existing}
    stats = RunStats(prior_spend_usd=sum(r["cost_usd"] for r in existing))
    budget = Budget(budget_usd, stats.prior_spend_usd)
    pending = [it for it in items if it["id"] not in done_ids]
    stats.skipped = len(done_ids)
    max_seen = 0.0
    consecutive_failures = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as out, ThreadPoolExecutor(max_workers=concurrency) as pool:
        inflight: dict[Future[Prediction], tuple[dict[str, Any], float]] = {}

        def drain(block_all: bool) -> None:
            nonlocal max_seen, consecutive_failures
            finished, _ = wait(inflight, return_when=ALL_COMPLETED if block_all else FIRST_COMPLETED)
            for fut in finished:
                item, est = inflight.pop(fut)
                try:
                    pred = fut.result()
                except Exception as exc:
                    budget.settle(est, 0.0)
                    stats.failed += 1
                    consecutive_failures += 1
                    log(f"error id={item['id']}: {type(exc).__name__}: {exc}")
                    continue
                consecutive_failures = 0
                budget.settle(est, pred.cost_usd)
                max_seen = max(max_seen, pred.cost_usd)
                record = {
                    "method": classifier.method,
                    "model": classifier.model,
                    "gold": item["label"],
                    "split": item["split"],
                    **pred.to_dict(),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                stats.done += 1

        for item in pending:
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log(f"aborting after {consecutive_failures} consecutive failures")
                break
            est = max(classifier.estimate_cost(item["text"]), max_seen)
            reserved = budget.try_reserve(est)
            if not reserved and inflight:
                drain(block_all=True)
                est = max(classifier.estimate_cost(item["text"]), max_seen)
                reserved = budget.try_reserve(est)
            if not reserved:
                stats.budget_stopped = True
                log(f"budget stop: spent ${budget.spent:.6f} + next ~${est:.6f} > ${budget.limit:.4f}")
                break
            fut = pool.submit(
                call_with_retries, lambda it=item: classifier.classify(it["id"], it["text"]), transient, sleep
            )
            inflight[fut] = (item, est)
            if len(inflight) >= concurrency:
                drain(block_all=False)
        if inflight:
            drain(block_all=True)

    stats.run_spend_usd = budget.spent - stats.prior_spend_usd
    log(
        f"done={stats.done} skipped={stats.skipped} failed={stats.failed} "
        f"spend: prior ${stats.prior_spend_usd:.6f} + run ${stats.run_spend_usd:.6f} = ${budget.spent:.6f} "
        f"(budget ${budget_usd:.4f})"
    )
    return stats


def build_classifier(args: argparse.Namespace) -> Classifier:
    from openai import OpenAI

    client = OpenAI(max_retries=0)
    if args.method == "decisions":
        from decision_triage.decisions_client import DecisionsClassifier

        return DecisionsClassifier(client, model=args.model or "gpt-6-luna")
    from decision_triage.responses_client import ResponsesClassifier

    effort = None if args.reasoning_effort == "omit" else args.reasoning_effort
    return ResponsesClassifier(client, model=args.model or "gpt-5.4-nano", reasoning_effort=effort)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one triage method over the eval set.")
    p.add_argument("--method", choices=["decisions", "responses"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument(
        "--reasoning-effort",
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "omit"],
        help="Responses only; 'omit' sends no reasoning parameter.",
    )
    p.add_argument("--data", type=Path, default=Path("data/eval.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--budget-usd", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    items = load_jsonl(args.data)
    if args.limit is not None:
        items = items[: args.limit]
    stats = run(
        build_classifier(args), items, args.out, concurrency=args.concurrency, budget_usd=args.budget_usd
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

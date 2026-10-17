"""Build data/eval.jsonl from public datasets (needs network and the "data" extra)."""

import argparse
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

SEED = 17
PER_QUEUE = 100
DEV_SHARE = 0.3

BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SMS = "ucirvine/sms_spam"

BITEXT_TO_QUEUE = {
    "INVOICE": "billing",
    "PAYMENT": "billing",
    "REFUND": "billing",
    "CANCEL": "billing",
    "ORDER": "orders",
    "DELIVERY": "orders",
    "SHIPPING": "orders",
    "ACCOUNT": "account",
    "SUBSCRIPTION": "account",
    "FEEDBACK": "feedback",
    "CONTACT": "human_agent",
}

Row = dict[str, str]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def bitext_candidates(rows: Iterable[dict]) -> list[Row]:
    out = []
    for r in rows:
        queue = BITEXT_TO_QUEUE.get(r["category"])
        if queue:
            out.append({
                "text": r["instruction"].strip(),
                "label": queue,
                "source": "bitext",
                "source_label": f"{r['category']}/{r['intent']}",
                "stratum": r["intent"],
            })
    return out


def sms_candidates(rows: Iterable[dict]) -> list[Row]:
    return [
        {
            "text": r["sms"].strip(),
            "label": "spam" if r["label"] == 1 else "other",
            "source": "sms_spam",
            "source_label": "spam" if r["label"] == 1 else "ham",
            "stratum": "",
        }
        for r in rows
    ]


def sample_queue(candidates: list[Row], n: int, rng: random.Random, seen: set[str]) -> list[Row]:
    """Round-robin over strata (intents) so each is represented as evenly as possible."""
    by_stratum: dict[str, list[Row]] = defaultdict(list)
    for c in candidates:
        by_stratum[c["stratum"]].append(c)
    pools = [by_stratum[k] for k in sorted(by_stratum)]
    for pool in pools:
        rng.shuffle(pool)
    picked: list[Row] = []
    while len(picked) < n and any(pools):
        for pool in pools:
            while pool:
                c = pool.pop()
                key = normalize(c["text"])
                if c["text"] and key not in seen:
                    seen.add(key)
                    picked.append(c)
                    break
            if len(picked) == n:
                break
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} unique rows for queue {candidates[0]['label']!r}")
    return picked


def build(candidates: list[Row], per_queue: int = PER_QUEUE, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    by_queue: dict[str, list[Row]] = defaultdict(list)
    for c in candidates:
        by_queue[c["label"]].append(c)
    seen: set[str] = set()
    records = []
    for queue in sorted(by_queue):
        picked = sample_queue(by_queue[queue], per_queue, rng, seen)
        rng.shuffle(picked)
        n_dev = round(DEV_SHARE * len(picked))
        for i, c in enumerate(picked):
            records.append({
                "id": f"{queue}-{i:03d}",
                "text": c["text"],
                "label": queue,
                "source": c["source"],
                "source_label": c["source_label"],
                "split": "dev" if i < n_dev else "test",
            })
    rng.shuffle(records)
    return records


def main() -> None:
    from datasets import load_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/eval.jsonl"))
    args = p.parse_args()

    bitext = load_dataset(BITEXT, split="train")
    missing = set(BITEXT_TO_QUEUE) - set(bitext.unique("category"))
    if missing:
        raise SystemExit(f"bitext categories missing: {sorted(missing)}")
    sms = load_dataset(SMS, split="train")
    records = build(bitext_candidates(bitext) + sms_candidates(sms))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        counts[(r["label"], r["split"])] += 1
    print(f"wrote {len(records)} rows to {args.out}")
    for (label, split), n in sorted(counts.items()):
        print(f"  {label:12s} {split:4s} {n}")


if __name__ == "__main__":
    main()

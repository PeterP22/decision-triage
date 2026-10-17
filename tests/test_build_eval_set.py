from collections import Counter

import build_eval_set as b


def fake_sources():
    bitext = []
    for cat in b.BITEXT_TO_QUEUE:
        for intent in ("a", "b", "c"):
            for k in range(12):
                bitext.append({
                    "instruction": f"  {cat} {intent} {k} about {{{{Order Number}}}}  ",
                    "category": cat,
                    "intent": f"{cat.lower()}_{intent}",
                })
    bitext.append({"instruction": "ORDER a 0 about {{order number}}", "category": "ORDER", "intent": "order_a"})
    sms = [{"sms": f"msg {i}\n", "label": i % 2} for i in range(60)]
    return b.bitext_candidates(bitext) + b.sms_candidates(sms)


def test_build_balanced_deduped_and_split():
    records = b.build(fake_sources(), per_queue=20)
    assert len(records) == 140
    expected = dict.fromkeys([*b.BITEXT_TO_QUEUE.values(), "spam", "other"], 20)
    assert Counter(r["label"] for r in records) == expected
    assert len({r["id"] for r in records}) == 140
    assert len({b.normalize(r["text"]) for r in records}) == 140
    splits = Counter((r["label"], r["split"]) for r in records)
    assert all(splits[(q, "dev")] == 6 and splits[(q, "test")] == 14 for q in {r["label"] for r in records})
    assert all(r["text"] == r["text"].strip() for r in records)
    assert any("{{Order Number}}" in r["text"] for r in records)
    assert set(records[0]) == {"id", "text", "label", "source", "source_label", "split"}


def test_build_stratifies_intents_and_is_deterministic():
    records = b.build(fake_sources(), per_queue=21)
    human = Counter(r["source_label"] for r in records if r["label"] == "human_agent")
    assert set(human.values()) == {7}
    assert records == b.build(fake_sources(), per_queue=21)

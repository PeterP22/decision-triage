# Evaluation data

`eval.jsonl` is built by `scripts/build_eval_set.py` (requires network and `pip install -e ".[data]"`).
It contains 700 messages: 100 per queue across 7 queues, split 30% dev / 70% test, stratified by queue.
Each line has `id`, `text`, `label` (the queue), `source`, `source_label` and `split`.

No real personal emails are used. Every message comes from one of the two public datasets below.

## Sources

| Queue | Source | Mapping |
|---|---|---|
| billing | Bitext | categories `INVOICE`, `PAYMENT`, `REFUND`, `CANCEL` |
| orders | Bitext | `ORDER`, `DELIVERY`, `SHIPPING` |
| account | Bitext | `ACCOUNT`, `SUBSCRIPTION` |
| feedback | Bitext | `FEEDBACK` |
| human_agent | Bitext | `CONTACT` |
| spam | SMS Spam Collection | `label == 1` (spam) |
| other | SMS Spam Collection | `label == 0` (ham) |

- **Bitext customer support dataset**: [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
  by Bitext Innovations. These are synthetic customer support utterances, licensed under the
  [Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)](https://cdla.dev/sharing-1-0/).
  Uses the `instruction` column. Template placeholders such as `{{Order Number}}` are left exactly as published.
  `source_label` is `CATEGORY/intent`.
- **SMS Spam Collection**: [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam),
  by Tiago A. Almeida and José María Gómez Hidalgo, distributed through the UCI Machine Learning Repository and licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Uses the `sms` column. `source_label` is `spam` or `ham`.
  Citation: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* DocEng 2011.

Because the Bitext rows are shared under CDLA-Sharing-1.0, `eval.jsonl` is redistributed under the same terms, with the
SMS rows also subject to CC BY 4.0 attribution.

## Sampling

- Seed 17, 100 rows per queue.
- For Bitext queues, rows are drawn round-robin across intents so each intent within a queue is represented as evenly as possible.
- Leading and trailing whitespace is stripped. Duplicates are removed across the whole set (case-insensitive, whitespace-collapsed).
- For each queue, 30 rows go to `dev` (used only to choose the confidence threshold) and 70 to `test` (used for all reported metrics).

## Caveats

- Bitext utterances are short, templated and synthetic. SMS messages are a different channel than email.
  Both sources make the task easier than real inbox traffic, so absolute accuracy here overstates production accuracy.
- `spam` and `other` come from one source and the other five queues from another. A model can partly separate them by
  style alone (SMS vs customer support phrasing).

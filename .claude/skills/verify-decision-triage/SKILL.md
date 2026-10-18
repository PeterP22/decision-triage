---
name: verify-decision-triage
description: "Prove decision-triage still works end to end: the report CLI, the rendered README, and the run_eval budget guard. Use before calling any change to src/, scripts/, results/ or README.template.md done."
---

# Verify decision-triage

One command drives the real CLIs and checks the properties that matter. No API key or network needed.

## Launch

```sh
pip install -e ".[dev]"        # once
python scripts/verify.py       # evidence goes to /tmp/verify-decision-triage
python scripts/verify.py DIR   # or to DIR
```

There is no server. Each run is a fresh set of CLI calls; two runs can share a box if they use different evidence dirs.

## Doctor

`python -c "import decision_triage, matplotlib, openai"` exits 0. If not, the install is broken: fix that before reading any FAIL.

## What it checks

| Check | User path | Proves |
|---|---|---|
| report reproduces `results/summary.json` | `python -m decision_triage.report --summary <dir>/summary.json --docs <dir>/docs` | the committed numbers come from the committed raw runs (floats equal to 1e-9) |
| report writes both charts | same run | `coverage.png` and `reliability.png` are written and non-trivial |
| README matches | `python scripts/render_readme.py --summary <dir>/summary.json --check` | every README number renders from the reproduced summary |
| budget guard | `python -m decision_triage.run_eval --method decisions --budget-usd 0 --limit 5` with `OPENAI_BASE_URL` pointed at a dead port | it stops before sending anything: exit 0, `budget stop`, `failed=0`, zero rows. Any call that slips through hits the dead port and shows up as `failed=N` |

## Evidence

Each check writes the exact command, exit code, stdout and stderr to `<dir>/*.log`, plus `<dir>/summary.json` and `<dir>/docs/*.png`. Cite the dir and the PASS lines in the Verification report.

## Cleanup

Nothing to tear down. The script never writes inside the repo; delete the evidence dir only after the PR is reviewed.

## Gotchas

- The README check renders percentages to one decimal, so a tiny metric drift can pass it. The summary check is the strict one.
- A real eval run costs money and needs `OPENAI_API_KEY`. This skill never makes one; ask Peter first.
- Changed numbers on purpose (new raw runs)? Re-run `python -m decision_triage.report`, then `python scripts/render_readme.py`, commit both, then run this.

Generated with peter-stack `playbooks/verification-skill.md`, adapted from pstack's create-verification-skill (MIT, Lauren Tan).

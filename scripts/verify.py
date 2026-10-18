"""Drive decision-triage the way a user does and check the properties that matter.

Usage: python scripts/verify.py [evidence-dir]   (default /tmp/verify-decision-triage)
Prints PASS/FAIL per check, exits 1 if any check fails. Evidence stays in the evidence dir.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEAD_API = "http://127.0.0.1:9/v1"
failed = False


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail and not ok else ""))
    failed |= not ok


def run(args: list[str], log: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env)
    log.write_text(f"$ {' '.join(args)}\nexit {result.returncode}\n"
                   f"--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}")
    return result


def close(a: object, b: object) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, int | float) and isinstance(b, int | float) and math.isclose(a, b, rel_tol=1e-9)
    return a == b


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/verify-decision-triage").resolve()
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    summary = out / "summary.json"
    r = run([py, "-m", "decision_triage.report", "--summary", str(summary), "--docs", str(out / "docs")],
            out / "report.log")
    check("report runs on committed raw results", r.returncode == 0, r.stderr.strip()[-200:])
    if r.returncode == 0:
        committed = json.loads((ROOT / "results/summary.json").read_text())
        check("report reproduces results/summary.json", close(json.loads(summary.read_text()), committed),
              f"diff {summary} results/summary.json")
        for png in ("coverage.png", "reliability.png"):
            path = out / "docs" / png
            check(f"report writes docs/{png}", path.exists() and path.stat().st_size > 10_000)

        r = run([py, "scripts/render_readme.py", "--summary", str(summary), "--check"], out / "readme.log")
        check("README numbers match the reproduced summary", r.returncode == 0, r.stderr.strip()[-200:])

    rows = out / "budget-zero.jsonl"
    rows.unlink(missing_ok=True)
    env = {**os.environ, "OPENAI_API_KEY": "verify-dummy", "OPENAI_BASE_URL": DEAD_API}
    r = run([py, "-m", "decision_triage.run_eval", "--method", "decisions", "--out", str(rows),
             "--budget-usd", "0", "--limit", "5"], out / "budget.log", env)
    sent = len(rows.read_text().splitlines()) if rows.exists() else 0
    check("run_eval --budget-usd 0 stops before any API call",
          r.returncode == 0 and "budget stop" in r.stderr and "failed=0" in r.stderr and sent == 0,
          f"exit {r.returncode}, rows {sent}, see {out / 'budget.log'}")

    print(f"evidence: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

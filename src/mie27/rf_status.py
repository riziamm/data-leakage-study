"""Random-forest deadline rule (DECISIONS.md D10).

Spec: RF runs on Gadi; if it cannot finish by 29 Oct 2026 18:00 (AEDT), drop it
and log the decision.

    python -m mie27.rf_status --run-id v1            # status only
    python -m mie27.rf_status --run-id v1 --log      # after the deadline: append the
                                                     # decision to DECISIONS.md

RF counts as finished only when every grid cell has >= --reps replicates.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .run import _cell_path, all_cells

TZ = ZoneInfo("Australia/Sydney")
DEADLINE = datetime(2026, 10, 29, 18, 0, tzinfo=TZ)


def rf_progress(run_dir: Path, reps: int) -> tuple[int, int]:
    cells = all_cells()
    done = 0
    for c in cells:
        p = _cell_path(run_dir, "rf", c)
        if p.exists() and len(pd.read_csv(p, usecols=["rep"])) >= reps:
            done += 1
    return done, len(cells)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--log", action="store_true", help="append the decision to DECISIONS.md")
    a = ap.parse_args(argv)
    now = datetime.now(TZ)
    done, total = rf_progress(Path(a.runs_dir) / a.run_id, a.reps)
    complete = done == total
    past = now >= DEADLINE
    print(f"{now:%Y-%m-%d %H:%M %Z}: RF cells complete {done}/{total} "
          f"(deadline {DEADLINE:%d %b %H:%M %Z}, {'passed' if past else 'not yet passed'})")
    if complete:
        decision = f"RF KEPT: all {total} cells complete at {now:%Y-%m-%d %H:%M %Z}."
        cmd = f"python -m mie27.summarise --run-id {a.run_id}"
    elif past:
        decision = (f"RF DROPPED under the deadline rule: {done}/{total} cells complete at "
                    f"{now:%Y-%m-%d %H:%M %Z}. Results, tests and figures use LR and kNN only; "
                    f"partial RF cell files are kept but not summarised.")
        cmd = f"python -m mie27.summarise --run-id {a.run_id} --models lr knn"
    else:
        print("No decision yet: RF incomplete and the deadline has not passed.")
        return
    print(decision)
    print("next:", cmd)
    if a.log:
        log = Path(__file__).resolve().parents[2] / "DECISIONS.md"
        with log.open("a") as f:
            f.write(f"\n### D10 decision ({now:%Y-%m-%d %H:%M %Z}, run {a.run_id})\n{decision}\n"
                    f"Summarised with: `{cmd}`\n")
        print(f"logged to {log}")


if __name__ == "__main__":
    main()

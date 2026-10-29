#!/usr/bin/env bash
# RF progress + projected finish (AEDT). Place next to run_all.sh; run: bash rf_progress.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
RUN_ID=${RUN_ID:-v1}; CHUNKS=${CHUNKS:-10}
START=$(grep "START grid_rf" run_all.log | tail -1 | cut -c1-19 || true)
[ -z "$START" ] && { echo "RF has not started yet (it starts after LR+kNN finish)."; exit 0; }
cd . && . .venv/bin/activate
python - "runs/${RUN_ID}-rf/cells/rf" "$CHUNKS" "$START" <<'PY'
import sys, glob, os, hashlib, datetime as dt
from zoneinfo import ZoneInfo
from mie27.run import all_cells
d, C, start = sys.argv[1], int(sys.argv[2]), sys.argv[3]
K = {"S2_k8": 8, "S3": 5, "S2_k4": 4, "S1_r2_d0": 4, "S1_r2_d0.2": 4}
cost = {hashlib.md5(c["cell_id"].encode()).hexdigest(): c["n"] * K.get(c["scenario"], 2) * c["p"] for c in all_cells()}
total_tasks, total_w = len(cost) * C, sum(cost.values()) * C
done = glob.glob(f"{d}/*.csv")
w = sum(cost[os.path.basename(f).split(".")[0]] for f in done)
t0 = dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S").astimezone()
now = dt.datetime.now().astimezone(); el = (now - t0).total_seconds()
syd = ZoneInfo("Australia/Sydney")
print(f"RF chunks done: {len(done)}/{total_tasks} ({len(done)/total_tasks:.1%}); cells fully done: "
      f"{sum(1 for h in cost if len(glob.glob(f'{d}/{h}.c*of{C}.csv')) == C)}/{len(cost)}")
print(f"cost-weighted progress: {w/total_w:.1%} after {el/3600:.2f} h")
if w:
    eta = t0 + dt.timedelta(seconds=el * total_w / w)
    print(f"projected finish: {eta.astimezone(syd):%a %d %b %H:%M %Z} "
          f"(slowest cells run first, so early projections are pessimistic)")
    dl = dt.datetime(2026, 10, 29, 18, 0, tzinfo=syd)
    print("PAST 18:00 AEDT 29 Oct" if eta > dl else "before 18:00 AEDT 29 Oct")
PY

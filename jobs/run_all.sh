#!/usr/bin/env bash
# ARCHIVED: ran the v1 grid from the parent directory; kept as a record, not runnable from jobs/.
# =============================================================================
# mie27-unit-leakage: one-shot, crash-recoverable driver (env -> grid -> tests).
#
# Put this file next to mie27-unit-leakage-src.tgz (no git history inside) and run:
#     nohup bash run_all.sh > run_all.out 2>&1 &
# Re-running after any crash/kill resumes: finished steps are skipped via markers in
# .state/, and the simulation itself resumes at replicate level inside each cell.
#
# Required once:  GIT_NAME="Your Name"  GIT_EMAIL="you@uni.edu"   (repo identity; no commits are made)
# Tunables (env vars):  N=<workers>  RAM_PER_WORKER_GB=0.25  RF=1|0  CHUNKS=10
#                       RUN_ID=v1  RETRIES=5  PY=python3.11 (fallback interpreter)
# =============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT=$PWD
REPO=$ROOT/mie27-unit-leakage
STATE=$ROOT/.state; mkdir -p "$STATE"
LOG=$ROOT/run_all.log
RUN_ID=${RUN_ID:-v1}; RF=${RF:-1}; CHUNKS=${CHUNKS:-10}; RETRIES=${RETRIES:-5}
RAM_PER_WORKER_GB=${RAM_PER_WORKER_GB:-0.25}

log()  { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
done_() { [ -f "$STATE/$1.done" ]; }
mark() { touch "$STATE/$1.done"; log "DONE  $1"; }
step() {  # step NAME CMD...   (skips if done, retries on failure, marks on success)
  local name=$1; shift
  if done_ "$name"; then log "SKIP  $name (already done)"; return 0; fi
  local try
  for try in $(seq 1 "$RETRIES"); do
    log "START $name (attempt $try/$RETRIES): $*"
    if "$@" >> "$LOG" 2>&1; then mark "$name"; return 0; fi
    log "FAIL  $name (attempt $try)"; sleep $((30 * try))
  done
  log "ABORT $name after $RETRIES attempts — see $LOG"; exit 1
}
trap 'log "signal/exit caught; rerun this script to resume"' INT TERM

# ---------------------------------------------------------------- 0. unpack repo
unpack() {
  if [ ! -d "$REPO" ]; then
    [ -f "$ROOT/mie27-unit-leakage-src.tgz" ] || { log "mie27-unit-leakage-src.tgz not found next to run_all.sh"; return 1; }
    tar xzf "$ROOT/mie27-unit-leakage-src.tgz" -C "$ROOT"
  fi
  # fresh repo under YOUR identity; nothing is committed — you commit when the paper is ready
  if [ ! -d "$REPO/.git" ]; then
    : "${GIT_NAME:?set GIT_NAME}" "${GIT_EMAIL:?set GIT_EMAIL}"
    git -C "$REPO" init -q -b sim-v1
    git -C "$REPO" config user.name "$GIT_NAME"
    git -C "$REPO" config user.email "$GIT_EMAIL"
  fi
}
step unpack unpack
cd "$REPO"

# ------------------------------------------------------------ 1. environment
make_env() {
  if [ -x .venv/bin/python ] && .venv/bin/python -c "import mie27, sklearn, matplotlib" 2>/dev/null; then return 0; fi
  rm -rf .venv
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
  if command -v uv >/dev/null; then
    uv venv -q -p 3.11 .venv && uv pip install -q --python .venv/bin/python -r requirements.lock \
      && uv pip install -q --python .venv/bin/python -e . && return 0
    log "uv path failed; trying plain venv"
  fi
  local py=${PY:-python3.11}
  command -v "$py" >/dev/null || { command -v module >/dev/null && module load python3/3.11 2>/dev/null; }
  "$py" -m venv .venv && .venv/bin/pip install -q -r requirements.lock && .venv/bin/pip install -q -e .
}
step env make_env
export PATH="$REPO/.venv/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import sys; print('python', sys.version.split()[0])" | tee -a "$LOG"

# -------------------------------------------------------- 2. worker count
cores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
ram_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
safe=$(awk -v r="$ram_gb" -v w="$RAM_PER_WORKER_GB" 'BEGIN{printf "%d", (r-8)/w}')
N=${N:-$cores}; [ "$N" -gt "$safe" ] && { log "N=$N exceeds RAM-safe $safe (avail ${ram_gb} GB); using $safe"; N=$safe; }
[ "$N" -gt "$cores" ] && N=$cores
log "cores=$cores availRAM=${ram_gb}GB workers=$N run_id=$RUN_ID chunks=$CHUNKS rf=$RF"

# ------------------------------------------------- 3. data-independent tests
step tests_static python -m pytest -q -p no:cacheprovider tests -k "test_3b or test_4 or test_5 or fast or repro"

# -------------------------------------------------------------- 4. LR + kNN
step grid_lrknn jobs/launch_local.sh "$RUN_ID" "$N" "lr knn"
verify_reps() {  # every expected (cell, model) must have 200 replicates
  python - "$@" <<'EOF'
import sys, glob, pandas as pd
run, models = sys.argv[1], sys.argv[2:]
from mie27.run import all_cells
bad = []
for m in models:
    files = glob.glob(f"runs/{run}/cells/{m}/*.csv")
    df = pd.concat(map(pd.read_csv, files)) if files else pd.DataFrame(columns=["cell_id", "rep"])
    n = df.groupby("cell_id").rep.nunique() if len(df) else pd.Series(dtype=int)
    missing = [c["cell_id"] for c in all_cells() if n.get(c["cell_id"], 0) < 200]
    bad += [f"{m}:{c}" for c in missing]
print(f"{len(bad)} incomplete (cell, model) pairs"); print("\n".join(bad[:20]))
sys.exit(1 if bad else 0)
EOF
}
step verify_lrknn verify_reps "$RUN_ID" lr knn

# -------------------------------------------------------------------- 5. RF
if [ "$RF" = 1 ]; then
  rf_launch() { CHUNKS=$CHUNKS jobs/launch_local.sh "${RUN_ID}-rf" "$N" "rf"; }
  step grid_rf rf_launch
  step verify_rf verify_reps "${RUN_ID}-rf" rf
  merge_rf() { mkdir -p "runs/$RUN_ID/cells/rf" && cp "runs/${RUN_ID}-rf/cells/rf/"*.csv "runs/$RUN_ID/cells/rf/"; }
  step merge_rf merge_rf
fi

# ----------------------------------------------------------- 6. post-process
step summarise python -m mie27.summarise --run-id "$RUN_ID"
run_tests() {
  MIE27_RUN_DIR="runs/$RUN_ID" python -m pytest -v -p no:cacheprovider tests 2>&1 | tee "runs/$RUN_ID/test_output.txt"
  return "${PIPESTATUS[0]}"
}
if ! done_ acceptance; then
  if run_tests >> "$LOG" 2>&1; then mark acceptance; else log "ACCEPTANCE TESTS FAILED — figures will be DRAFT only"; touch "$STATE/acceptance.failed"; fi
fi
figs() { if [ -f "$STATE/acceptance.done" ]; then python -m mie27.figures --run-id "$RUN_ID"; else python -m mie27.figures --run-id "$RUN_ID" --draft; fi; }
step figures figs
step calculator bash -c "python -m mie27.build_calculator --run-id '$RUN_ID' | tee 'runs/$RUN_ID/calculator_report.txt'"
step pr_body bash -c "{ echo '## Tests'; cat 'runs/$RUN_ID/test_output.txt'; echo; echo '## Calculator'; cat 'runs/$RUN_ID/calculator_report.txt'; echo; echo '## Calibration'; cat 'runs/$RUN_ID/calibration.csv'; echo; echo '## Code'; git status --short | wc -l | sed 's/^/uncommitted files: /'; } > 'runs/$RUN_ID/PR_BODY.md'"

log "ALL DONE. Outputs: $REPO/runs/$RUN_ID  (results.csv, fig1_nomogram*.pdf, fig2_null*.pdf, calculator.py, PR_BODY.md)"
[ -f "$STATE/acceptance.failed" ] && log "NOTE: acceptance tests failed; see runs/$RUN_ID/test_output.txt; figures are DRAFT. Fix, then: rm $STATE/{acceptance.failed,figures.done} and rerun." 
exit 0

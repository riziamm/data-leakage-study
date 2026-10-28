#!/bin/bash
# Run the grid on N local cores. Resumable: re-running skips finished replicates.
# Usage: jobs/launch_local.sh RUN_ID N "MODELS" [RUNS_DIR] [REPS]
#   jobs/launch_local.sh v1    250 "lr knn"
#   jobs/launch_local.sh v1-rf 250 "rf"
set -euo pipefail
RUN_ID=$1; N=$2; MODELS=$3; RUNS_DIR=${4:-runs}; REPS=${5:-200}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per worker
mkdir -p "$RUNS_DIR/$RUN_ID"
exec python -m mie27.run --run-id "$RUN_ID" --runs-dir "$RUNS_DIR" --models $MODELS \
     --reps "$REPS" --jobs "$N" --chunks "${CHUNKS:-1}" 2>&1 | tee -a "$RUNS_DIR/$RUN_ID/launch.log"

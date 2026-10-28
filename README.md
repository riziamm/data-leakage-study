# mie27-unit-leakage

Simulation for MIE 2027: how much does row-level cross-validation overstate AUC
when patients contribute several correlated rows (eyes, imaging visits, EHR
encounters)? **Fully synthetic — no real patient data exists anywhere in this
repository.** Python 3.11; every number is seeded and reproducible.

Read `DECISIONS.md` before using any output: it records where the spec was
ambiguous or impossible as written (S1 variance nesting, the AUC estimator,
acceptance test 3, random-forest deadline).

## Layout
| path | what |
|---|---|
| `src/mie27/generate.py` | generative model, scenarios control / S1 / S2 / S3 |
| `src/mie27/calibrate.py` | beta ↔ true-AUC calibration (analytic + Monte Carlo) |
| `src/mie27/cv.py` | splits (a) row, (b) unit, (c) patient; leakage guard; models |
| `src/mie27/run.py` | grid runner: per-cell seeds, resumable, shardable |
| `src/mie27/summarise.py` | → `runs/<run_id>/results.csv`, `calibration.csv` |
| `src/mie27/figures.py` | → `fig1_nomogram.pdf`, `fig2_null.pdf` (gated on tests) |
| `src/mie27/build_calculator.py` | → `runs/<run_id>/calculator.py` + fit error |
| `tests/` | acceptance tests 0–5 |
| `jobs/launch_local.sh` | run the grid on N local cores (resumable, chunkable) |

## Reproduce everything (250-core node)

```bash
# 0. environment (once) — Python 3.11, versions pinned in requirements.lock
uv venv -p 3.11 .venv && . .venv/bin/activate
uv pip install -r requirements.lock && uv pip install -e .

# 1. LR + kNN, full grid, 200 replicates   (~15 min on 250 cores, ~55 GB RAM)
jobs/launch_local.sh v1 250 "lr knn"
# 2. RF, separate run                      (~5-6 h on 250 cores)
CHUNKS=10 jobs/launch_local.sh v1-rf 250 "rf"
#    merge RF into v1 when done:  cp -r runs/v1-rf/cells/rf runs/v1/cells/

# 3. results.csv + calibration.csv
python -m mie27.summarise --run-id v1
# 4. acceptance tests 0-5 (writes runs/v1/acceptance.json)
MIE27_RUN_DIR=runs/v1 python -m pytest -v tests | tee runs/v1/test_output.txt
# 5. figures (refuse unless step 4 passed; carry run_id + git hash)
python -m mie27.figures --run-id v1
# 6. calculator + held-out validation
python -m mie27.build_calculator --run-id v1 | tee runs/v1/calculator_report.txt
```
`jobs/launch_local.sh RUN_ID N "MODELS" [RUNS_DIR] [REPS]`; `CHUNKS=k` splits each
cell's replicates across k tasks. Every run is resumable: rerun the same command.

## results.csv columns
One row per cell × model × split (`a_row`, `b_unit`, `c_patient`).
`auc_mean`, `auc_q025`, `auc_q975`: fold-mean CV AUC across replicates.
`optimism_mean` (+ `_mcse`, `_q025`, `_q975`): this split minus split (c); the
`a_row` row is the headline (a) − (c). `false_alarm_070`: P(AUC > 0.70) for
true AUC 0.50 cells. `*_pooled`: same with pooled out-of-fold AUC (sensitivity).
`runtime_s` (whole cell, that model), `seed`, `run_id`, `git_hash`, `k_mean`,
`skipped_folds_mean`, `redraws_total`.

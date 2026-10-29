#!/usr/bin/env bash
# Follow-up to the LR+kNN subset (approved 29 Oct). Place next to run_all.sh; first
# apply + commit test3_fix.patch (see instructions), then:
#     bash followup.sh 2>&1 | tee followup.out
# Safe while RF runs: does not touch src/, runs/v1/cells or runs/v1-rf; uses 16 cores.
set -euo pipefail
WORK="$(dirname "$(readlink -f "$0")")/.."
cd "$WORK/mie27-unit-leakage" && . .venv/bin/activate
RUN_ID=${RUN_ID:-v1}; JOBS=${JOBS:-16}; R=${REPS:-200}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f scripts/test3_comparator.py ] || { echo "apply test3_fix.patch first"; exit 1; }
[ -d "early/$RUN_ID" ] || { echo "run early_results.sh first"; exit 1; }

echo "== 1. run ids and code versions =="
python - "$RUN_ID" <<'EOF'
import sys, glob, pandas as pd
run = sys.argv[1]
for d in (f"runs/{run}/cells/lr", f"runs/{run}/cells/knn", f"runs/{run}-rf/cells/rf"):
    fs = glob.glob(f"{d}/*.csv")
    h = sorted({x for f in fs[:2000] for x in pd.read_csv(f, usecols=["git_hash"]).git_hash.astype(str)})
    print(f"  {d:28s} files={len(fs):5d} git_hash={h}")
print(f"  early subset: early/{run}  (copy of runs/{run} LR+kNN cells; run_id column = {run})")
EOF
echo "  HEAD now: $(git rev-parse HEAD) ($(git log -1 --format=%s))"

echo "== 2. test-3 comparator (80% training), LR + kNN now; RF after RF finishes =="
python scripts/test3_comparator.py --run-dir "runs/$RUN_ID" --models lr knn --jobs "$JOBS" --reps "$R" 2>&1 | grep -v "Parallel("
cp "runs/$RUN_ID/test3_comparator80.csv" "early/$RUN_ID/"
echo "-- tests 0-3 on the LR+kNN subset with the new comparator --"
MIE27_RUN_DIR="early/$RUN_ID" python -m pytest -v -p no:cacheprovider tests/test_acceptance.py \
    -k "test_0 or test_1 or test_2 or test_3_" 2>&1 | tee "early/$RUN_ID/tests_0_3_v2comparator_lr_knn_${RUN_ID}.txt" || true
[ -f "early/$RUN_ID/acceptance.json" ] && mv "early/$RUN_ID/acceptance.json" "early/$RUN_ID/acceptance_subset_v2comparator.json"
python - "early/$RUN_ID" <<'EOF'
import sys, pandas as pd
E = sys.argv[1]
r = pd.read_csv(f"{E}/results.csv"); c = pd.read_csv(f"{E}/test3_comparator80.csv")
d = r[(r.true_auc == .8) & (r.n == 200) & (r.icc == 0)].merge(c[["cell_id", "model", "auc_indep80_mean"]], on=["cell_id", "model"])
d["gap_80"] = d.auc_mean - d.auc_indep80_mean; d["gap_100"] = d.auc_mean - d.auc_indep_mean
pd.set_option("display.width", 200)
print("test-3 gaps, all rows (limit |gap_80| <= 0.02):")
print(d[["run_id", "model", "scenario", "p", "split", "auc_mean", "auc_indep80_mean", "gap_80", "gap_100"]]
      .sort_values("gap_80").to_string(index=False, float_format="%.4f"))
EOF

echo "== 3. control n=20 p=20 ICC 0.9 AUC 0.80: replicates 201-400 (run_id ${RUN_ID}-ctrl-ext) =="
python - "$RUN_ID" "$R" <<'EOF'
import sys, glob, hashlib, numpy as np, pandas as pd
from pathlib import Path
from mie27.run import all_cells, run_cell, run_replicate
run, R = sys.argv[1], int(sys.argv[2])
cell = all_cells(dict(scenario=["control"], n=[20], p=[20], icc=[0.9], true_auc=[0.8]))[0]
out = Path(f"runs/{run}-ctrl-ext")
# replicates R..2R-1 (1-based 201-400) = chunk 1 of 2 over 2R; same seeds as an uninterrupted 2R run
run_cell(cell, ["lr", "knn"], 2 * R, out, chunk=(1, 2))
h = hashlib.md5(cell["cell_id"].encode()).hexdigest()
# seed continuity check: replicate 0 under spawn(2R) must equal the stored replicate 0
r0 = run_replicate(cell, ["knn"], np.random.SeedSequence(cell["seed"]).spawn(2 * R)[0])["knn"]
orig = {m: pd.read_csv(f"runs/{run}/cells/{m}/{h}.csv") for m in ("lr", "knn")}
assert abs(r0["aucfm_row"] - orig["knn"].aucfm_row.iloc[0]) < 1e-12, "seed continuity broken"
print(f"  seed continuity verified (rep 1 recomputed under spawn({2*R}) == stored)")
print(f"  pre-stated reading: chance if new-{R} |z| < 2 and combined |z| < 3; "
      f"systematic if new-{R} z <= -2 (same sign as before)")
for m in ("knn", "lr"):
    new = pd.read_csv(glob.glob(f"{out}/cells/{m}/{h}.c1of2.csv")[0])
    assert new.rep.min() == R and new.rep.max() == 2 * R - 1
    for lab, df in (("reps 1-200 (v1)", orig[m]), (f"reps {R+1}-{2*R} (ext)", new),
                    (f"reps 1-{2*R} combined", pd.concat([orig[m], new]))):
        dd = df.aucfm_row - df.aucfm_patient
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"  {run}{'-ctrl-ext' if 'ext' in lab else ''} {m:4s} {lab:22s} optimism {dd.mean():+.4f} "
              f"(MC SE {se:.4f}, z {dd.mean()/se:+.2f}, n={len(dd)})")
EOF

echo "== 4. decomposition of the two max-inflation cells (run_id $RUN_ID) =="
python - "early/$RUN_ID" "$RUN_ID" <<'EOF'
import sys, numpy as np, pandas as pd
rep = pd.read_csv(f"{sys.argv[1]}/replicates.csv.gz"); run = sys.argv[2]
def ms(x):
    x = np.asarray(x, float); return f"{x.mean():+.4f} (SE {x.std(ddof=1)/np.sqrt(len(x)):.4f})"
for m, cid in (("knn", "S2_k8|n30|icc0.9|p20|auc0.50"), ("lr", "S3|n20|icc0.9|p180|auc0.50")):
    g = rep[(rep.model == m) & (rep.cell_id == cid)]
    a, b, c, ind = g.aucfm_row, g.aucfm_unit, g.aucfm_patient, g.auc_indep
    print(f"\n  {run} | {m} | {cid} | {len(g)} replicates | git {g.git_hash.iloc[0][:10]}")
    print(f"    AUC row (a)          {ms(a)}   2.5/50/97.5%: {np.percentile(a,[2.5,50,97.5]).round(3)}")
    print(f"    AUC unit (b)         {ms(b)}   (b = a outside S1)")
    print(f"    AUC patient (c)      {ms(c)}   2.5/50/97.5%: {np.percentile(c,[2.5,50,97.5]).round(3)}")
    print(f"    AUC independent test {ms(ind)}   (model trained on all rows; truth = 0.50)")
    print(f"    inflation a - c      {ms(a-c)}")
    print(f"      = leakage  a - indep        {ms(a-ind)}")
    print(f"      + honest-split bias indep - c {ms(ind-c)}")
    print(f"    pooled-AUC inflation          {ms(g.auc_row-g.auc_patient)}")
    print(f"    P(a > 0.6) {np.mean(a>.6):.3f}  P(a > 0.7) {np.mean(a>.7):.3f}  P(a > 0.9) {np.mean(a>.9):.3f}  "
          f"P(c > 0.7) {np.mean(c>.7):.3f}")
    print(f"    skipped single-class test folds per replicate: row {g.skipfold_row.mean():.2f}, "
          f"patient {g.skipfold_patient.mean():.2f}")
    print(f"    rows/patient per replicate: mean {g.k_mean.mean():.2f}, min {g.k_mean.min():.2f}, "
          f"max {g.k_mean.max():.2f}; rows mean {g.n_rows.mean():.0f}")
    if g.k_mean.nunique() > 1:
        print(f"    Spearman(inflation, rows/patient) = {pd.Series(a-c).corr(g.k_mean, method='spearman'):+.3f}")
EOF

echo "== 5. RF progress =="
bash "$WORK/rf_progress.sh" || true
echo "== done =="

#!/usr/bin/env bash
# Early LR + kNN results while RF runs. Place next to run_all.sh; run:
#     bash early_results.sh 2>&1 | tee early_results.out
# Works on a COPY of the LR/kNN cells in mie27-unit-leakage/early/<run_id>/, so it
# never touches runs/<run_id>/ (no results.csv, acceptance.json or figures there).
# Refuses to produce anything unless every LR and kNN cell has exactly reps 0..199.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
. .venv/bin/activate
RUN_ID=${RUN_ID:-v1}; MIN_REPS=${MIN_REPS:-200}; CHECK_FULL_GRID=${CHECK_FULL_GRID:-1}
SRC=runs/$RUN_ID; E=early/$RUN_ID; TAG=lr_knn_${RUN_ID}

echo "== 0. completeness gate (>= $MIN_REPS replicates, every cell) =="
python - "$SRC" "$MIN_REPS" "$CHECK_FULL_GRID" <<'EOF'
import sys, glob, pandas as pd
from mie27.run import all_cells
src, need, full = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
bad, hashes = [], set()
for m in ("lr", "knn"):
    files = glob.glob(f"{src}/cells/{m}/*.csv")
    df = pd.concat(map(pd.read_csv, files)) if files else pd.DataFrame(columns=["cell_id", "rep", "git_hash"])
    hashes |= set(df.git_hash.astype(str))
    reps = df.groupby("cell_id").rep.apply(lambda r: sorted(r) == list(range(len(r))) and len(r) >= need)
    cnt = df.groupby("cell_id").rep.nunique()
    ids = [c["cell_id"] for c in all_cells()] if full else list(cnt.index)
    for c in ids:
        if not reps.get(c, False):
            bad.append((m, c, int(cnt.get(c, 0))))
print(f"git_hash values in cells: {sorted(hashes)}")
if bad:
    print(f"NOT READY: {len(bad)} (model, cell) pairs below {need} replicates")
    for m, c, k in bad[:50]:
        print(f"  {m:4s} {c:45s} {k} reps")
    sys.exit(1)
if len(hashes) > 1 or any(h.endswith("-dirty") for h in hashes):
    print("WARNING: cells come from more than one code version or a dirty tree")
print("OK: all LR and kNN cells complete")
EOF

echo "== 1. summarise LR + kNN subset =="
rm -rf "$E"; mkdir -p "$E/cells"
cp -r "$SRC/cells/lr" "$SRC/cells/knn" "$E/cells/"
python -m mie27.summarise --run-id "$RUN_ID" --runs-dir early --models lr knn
cp "$E/results.csv" "$E/results_${TAG}.csv"
echo "wrote $E/results_${TAG}.csv"

echo "== 2. tests 0-3 on the subset =="
MIE27_RUN_DIR="$E" python -m pytest -v -p no:cacheprovider tests/test_acceptance.py \
    -k "test_0 or test_1 or test_2 or test_3" 2>&1 | tee "$E/tests_0_3_${TAG}.txt" || true
[ -f "$E/acceptance.json" ] && mv "$E/acceptance.json" "$E/acceptance_subset_${TAG}.json"

echo "== 2b. every cell with |z| >= 3 (tests 1 and 2 scope), and failing fraction =="
python - "$E/results.csv" <<'EOF' | tee "$E/zscores_${TAG}.txt"
import sys, pandas as pd
r = pd.read_csv(sys.argv[1]); a = r[r.split == "a_row"].copy()
a["z"] = a.optimism_mean / a.optimism_mcse
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
cols = ["run_id", "model", "scenario", "n", "p", "icc", "true_auc", "optimism_mean", "optimism_mcse", "z", "n_reps"]
for name, sub in [("test 1: ICC = 0", a[a.icc == 0]), ("test 2: control k = 1", a[a.scenario == "control"])]:
    for m, g in sub.groupby("model"):
        bad = g[g.z.abs() >= 3]
        print(f"\n{name} | {m}: {len(bad)}/{len(g)} cells with |z| >= 3 = {len(bad)/len(g):.2%} (gate <= 1%); "
              f"|optimism| >= 0.01 in {int((g.optimism_mean.abs() >= .01).sum())} cells (reported only)")
        if len(bad):
            print(bad[cols].to_string(index=False))
EOF

echo "== 3. DRAFT figures from the subset =="
python -m mie27.figures --run-id "$RUN_ID" --runs-dir early --draft
mv "$E/fig1_nomogram_DRAFT.pdf" "$E/fig1_nomogram_${TAG}_DRAFT.pdf"
mv "$E/fig2_null_DRAFT.pdf"     "$E/fig2_null_${TAG}_DRAFT.pdf"
echo "wrote $E/fig1_nomogram_${TAG}_DRAFT.pdf and $E/fig2_null_${TAG}_DRAFT.pdf"

echo "== 4. headline numbers =="
python - "$E" <<'EOF' | tee "$E/numbers_${TAG}.txt"
import sys, numpy as np, pandas as pd
E = sys.argv[1]
r = pd.read_csv(f"{E}/results.csv"); rep = pd.read_csv(f"{E}/replicates.csv.gz")
run = r.run_id.iloc[0]; a = r[r.split == "a_row"]
assert r.n_reps.min() >= 200 or "MIN_REPS" in __import__("os").environ, "fewer than 200 replicates"
print(f"run_id {run} | git {r.git_hash.iloc[0]} | replicates/cell min {r.n_reps.min()} max {r.n_reps.max()}")

print("\n(a) largest row-split minus patient-split inflation, per classifier")
for m, g in a.groupby("model"):
    t = g.loc[g.optimism_mean.idxmax()]
    print(f"  {run} {m:4s} {t.scenario:11s} n={t.n:<3d} p={t.p:<3d} ICC={t.icc:<3g} true AUC={t.true_auc:.2f}: "
          f"{t.optimism_mean:+.4f} (MC SE {t.optimism_mcse:.4f})")

print("\n(b) null false-alarm rate, row split, all true-AUC 0.50 cells pooled")
nul = rep[rep.true_auc == 0.5]
for m, g in nul.groupby("model"):
    for thr in (0.6, 0.7):
        per_cell = g.groupby("cell_id").aucfm_row.apply(lambda x: np.mean(x > thr))
        se = per_cell.std(ddof=1) / np.sqrt(len(per_cell))
        frac_cells = np.mean(g.groupby("cell_id").aucfm_row.mean() > thr)
        print(f"  {run} {m:4s} all scenarios/n/p/ICC: P(AUC_a > {thr}) = {per_cell.mean():.4f} "
              f"(SE across {len(per_cell)} cells {se:.4f}); fraction of cells with mean AUC_a > {thr}: {frac_cells:.4f}")

print("\n(c) inflation at ICC = 0, row minus patient, per classifier (pooled over all ICC = 0 cells)")
for m, g in a[a.icc == 0].groupby("model"):
    est = g.optimism_mean.mean(); se = np.sqrt((g.optimism_mcse ** 2).sum()) / len(g)
    z = g.optimism_mean / g.optimism_mcse
    print(f"  {run} {m:4s} all scenarios, n, p, AUC at ICC=0 ({len(g)} cells): {est:+.5f} (MC SE {se:.5f}, "
          f"z {est/se:+.2f}); per-cell max |z| {z.abs().max():.2f}")

print("\n(d) pooled OOF minus fold-mean AUC at ICC = 0, per classifier and split")
z0 = rep[rep.icc == 0]
for m, g in z0.groupby("model"):
    for sp in ("row", "patient"):
        d = (g[f"auc_{sp}"] - g[f"aucfm_{sp}"])
        cell = d.groupby(g.cell_id).mean()
        se = cell.std(ddof=1) / np.sqrt(len(cell))
        print(f"  {run} {m:4s} split={sp:7s} all ICC=0 cells ({len(cell)}): {d.mean():+.5f} (SE across cells {se:.5f}); "
              f"most negative cell {cell.min():+.4f} [{cell.idxmin()}]")
EOF
echo "== done: everything is in mie27-unit-leakage/$E =="

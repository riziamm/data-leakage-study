#!/usr/bin/env bash
# Final v1 report (LR + kNN + RF). Place next to run_all.sh; run AFTER run_all.sh
# has finished with all acceptance tests passing:
#     bash final_report.sh 2>&1 | tee final_report.out
# Writes runs/v1/figs_v2/* and runs/v1/final_numbers.txt. Refuses if acceptance failed.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." && . .venv/bin/activate
RUN_ID=${RUN_ID:-v1}; R=runs/$RUN_ID

echo "== 0. gate =="
python - "$R" <<'EOF'
import sys, json, pandas as pd
R = sys.argv[1]
acc = json.load(open(f"{R}/acceptance.json"))
bad = {k: v for k, v in acc["outcomes"].items() if v != "passed"}
print(f"acceptance.json: all_passed={acc['all_passed']} ({len(acc['outcomes'])} tests, {acc['when']})")
for k, v in bad.items(): print("  NOT PASSED:", v, k)
r = pd.read_csv(f"{R}/results.csv")
print(f"results.csv: models={sorted(r.model.unique())} cells/model={r.groupby('model').cell_id.nunique().to_dict()} "
      f"reps min={r.n_reps.min()} git={sorted(r.git_hash.astype(str).unique())}")
ok = acc["all_passed"] and set(r.model) == {"lr", "knn", "rf"} and r.n_reps.min() >= 200 \
     and (r.groupby("model").cell_id.nunique() == 960).all()
sys.exit(0 if ok else 1)
EOF

echo "== 1. manuscript figures (PNG 600 dpi, SVG, PDF) =="
python scripts/figures_v2.py --results "$R/results.csv" --outdir "$R/figs_v2" \
       --models lr knn rf --note "LR, kNN, RF" | tee "$R/figs_v2_report.txt"

echo "== 2. headline numbers =="
python - "$R" <<'EOF' | tee "$R/final_numbers.txt"
import sys, numpy as np, pandas as pd
R = sys.argv[1]
r = pd.read_csv(f"{R}/results.csv"); rep = pd.read_csv(f"{R}/replicates.csv.gz")
run = r.run_id.iloc[0]; a = r[r.split == "a_row"]
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
print(f"run_id {run} | git {r.git_hash.iloc[0]} | replicates/cell {r.n_reps.min()}-{r.n_reps.max()} | "
      f"models {sorted(r.model.unique())}")

print("\n(T) tests 1-2 scope: cells with |z| >= 3 (gate <= 1%)")
a2 = a.assign(z=a.optimism_mean / a.optimism_mcse)
for name, sub in [("test 1 ICC=0", a2[a2.icc == 0]), ("test 2 control", a2[a2.scenario == "control"])]:
    for m, g in sub.groupby("model"):
        b = g[g.z.abs() >= 3]
        print(f"  {name:15s} {m:4s} {len(b)}/{len(g)} = {len(b)/len(g):.2%}; |opt|>=0.01 in "
              f"{int((g.optimism_mean.abs() >= .01).sum())}")
        if len(b): print(b[["model", "scenario", "n", "p", "icc", "true_auc", "optimism_mean",
                            "optimism_mcse", "z"]].to_string(index=False))

print("\n(a) largest row minus patient inflation, per classifier")
for m, g in a.groupby("model"):
    t = g.loc[g.optimism_mean.idxmax()]
    print(f"  {run} {m:4s} {t.scenario:11s} n={t.n:<3d} p={t.p:<3d} ICC={t.icc:<3g} AUC={t.true_auc:.2f}: "
          f"{t.optimism_mean:+.4f} (MC SE {t.optimism_mcse:.4f})")

print("\n(b) null false alarm, row split, pooled over 320 no-signal cells (grid-composition dependent)")
nul = rep[rep.true_auc == 0.5]
for m, g in nul.groupby("model"):
    for thr in (0.6, 0.7):
        pc = g.groupby("cell_id").aucfm_row.apply(lambda x: np.mean(x > thr))
        print(f"  {run} {m:4s} P(AUC_a > {thr}) = {pc.mean():.4f} (between-cell SE {pc.std(ddof=1)/np.sqrt(len(pc)):.4f})")

print("\n(c) inflation at ICC = 0, pooled over 240 cells")
for m, g in a[a.icc == 0].groupby("model"):
    est = g.optimism_mean.mean(); se = np.sqrt((g.optimism_mcse ** 2).sum()) / len(g)
    print(f"  {run} {m:4s} {est:+.5f} (MC SE {se:.5f}, z {est/se:+.2f}); per-cell max |z| "
          f"{(g.optimism_mean/g.optimism_mcse).abs().max():.2f}")

print("\n(d) pooled OOF minus fold-mean AUC at ICC = 0")
z0 = rep[rep.icc == 0]
for m, g in z0.groupby("model"):
    for sp in ("row", "patient"):
        cell = (g[f"auc_{sp}"] - g[f"aucfm_{sp}"]).groupby(g.cell_id).mean()
        print(f"  {run} {m:4s} {sp:7s} {cell.mean():+.5f} (SE {cell.std(ddof=1)/np.sqrt(len(cell)):.5f})")

print("\n(e) decomposition of each classifier's max-inflation cell: a - c = (a - indep) + (indep - c)")
for m, g in a.groupby("model"):
    cid = g.loc[g.optimism_mean.idxmax(), "cell_id"]
    x = rep[(rep.model == m) & (rep.cell_id == cid)]
    A, C, I = x.aucfm_row, x.aucfm_patient, x.auc_indep
    se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
    print(f"  {run} {m:4s} {cid}: a {A.mean():.3f}  c {C.mean():.3f}  indep {I.mean():.3f} | "
          f"inflation {np.mean(A-C):+.4f} ({se(A-C):.4f}) = leakage {np.mean(A-I):+.4f} ({se(A-I):.4f}) "
          f"+ honest-split bias {np.mean(I-C):+.4f} ({se(I-C):.4f})")

print("\n(f) test 3 (80%-trained comparator): max |gap| per classifier")
c = pd.read_csv(f"{R}/test3_comparator80.csv")
d = r[(r.true_auc == .8) & (r.n == 200) & (r.icc == 0)].merge(c[["cell_id", "model", "auc_indep80_mean"]], on=["cell_id", "model"])
for m, g in d.groupby("model"):
    gap = g.auc_mean - g.auc_indep80_mean
    print(f"  {run} {m:4s} max |gap| {gap.abs().max():.4f} (limit 0.02)")

print("\n(g) runtime, core-hours by classifier")
print("  " + ", ".join(f"{m} {v/3600:.1f}" for m, v in r[r.split == "a_row"].groupby("model").runtime_s.sum().items()))
EOF

echo "== 3. calculator =="
cat "$R/calculator_report.txt"
echo "== done: figures in $R/figs_v2, numbers in $R/final_numbers.txt =="

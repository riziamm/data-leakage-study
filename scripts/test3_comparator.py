"""Test-3 comparator (approved 29 Oct): independent-test AUC of a model trained on
80% of patients, matching the CV training size, for the 16 test-3 cells
(true AUC 0.80, n = 200, ICC = 0, all scenarios, p in {20, 180}).

Each replicate is regenerated exactly from its seed (same SeedSequence children as
mie27.run.run_replicate). Regeneration is verified against the stored full-data
auc_indep before anything is written; the 80% subsample uses a 4th, previously
unused child of the replicate SeedSequence. Does not touch src/ or the grid.

    python scripts/test3_comparator.py --run-dir runs/v1 --rf-cells runs/v1-rf/cells/rf --jobs 16
"""
from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from mie27.calibrate import beta_for
from mie27.cv import make_model
from mie27.generate import simulate
from mie27.run import N_INDEP, N_REPEATS, all_cells

REPS, FRAC = 200, 0.8


def stored_indep(cells_dir: str, model: str, cid: str) -> pd.Series | None:
    h = hashlib.md5(cid.encode()).hexdigest()
    files = glob.glob(f"{cells_dir}/{model}/{h}*.csv") or glob.glob(f"{cells_dir}/{h}*.csv")
    if not files:
        return None
    df = pd.concat(map(pd.read_csv, files))
    return df.set_index("rep").auc_indep.sort_index()


def one_task(cell: dict, model: str, stored: pd.Series | None, n_check: int) -> dict:
    beta = beta_for(cell["true_auc"], cell["p"])
    reps = np.random.SeedSequence(cell["seed"]).spawn(REPS)
    a80, afull, diffs = [], [], []
    for r, rep_ss in enumerate(reps):
        data_ss, cv_ss, indep_ss, sub_ss = rep_ss.spawn(4)        # first 3 == run_replicate
        d = simulate(cell["scenario"], cell["n"], cell["icc"], cell["p"], beta,
                     np.random.default_rng(data_ss))
        cv_rng = np.random.default_rng(cv_ss)
        cv_rng.integers(0, 2**31 - 1, size=N_REPEATS)              # cv_seeds, consumed as in run
        model_seed = int(cv_rng.integers(0, 2**31 - 1))
        d_ind = simulate("control", N_INDEP, cell["icc"], cell["p"], beta,
                         np.random.default_rng(indep_ss))
        # regeneration check: full-data model must reproduce the stored auc_indep
        if stored is not None and r < n_check and r in stored.index:
            full = make_model(model, model_seed).fit(d["X"], d["y"])
            af = roc_auc_score(d_ind["y"], full.predict_proba(d_ind["X"])[:, 1])
            afull.append(af)
            diffs.append(abs(af - stored.loc[r]))
        # 80% of patients, stratified by patient label
        srng = np.random.default_rng(sub_ss)
        yp = d["y_patient"]
        keep = np.concatenate([srng.choice(np.flatnonzero(yp == c), int(round(FRAC * (yp == c).sum())),
                                           replace=False) for c in (0, 1)])
        rows = np.isin(d["patient"], keep)
        m = make_model(model, model_seed).fit(d["X"][rows], d["y"][rows])
        a80.append(roc_auc_score(d_ind["y"], m.predict_proba(d_ind["X"])[:, 1]))
    a80 = np.array(a80)
    return dict(cell_id=cell["cell_id"], scenario=cell["scenario"], n=cell["n"], icc=cell["icc"],
                p=cell["p"], true_auc=cell["true_auc"], model=model, n_reps=len(a80),
                auc_indep80_mean=a80.mean(), auc_indep80_mcse=a80.std(ddof=1) / np.sqrt(len(a80)),
                regen_checked=len(diffs), regen_max_abs_diff=max(diffs) if diffs else np.nan,
                regen_ok=bool(diffs) and max(diffs) < 1e-9)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/v1")
    ap.add_argument("--rf-cells", default="runs/v1-rf/cells/rf")
    ap.add_argument("--models", nargs="+", default=["lr", "knn", "rf"])
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--rf-check", type=int, default=5, help="replicates to regen-check for RF")
    ap.add_argument("--reps", type=int, default=200, help="keep 200 except for smoke tests")
    ap.add_argument("--scenario", nargs="*", default=None, help="smoke tests only")
    ap.add_argument("--p", nargs="*", type=int, default=None, help="smoke tests only")
    a = ap.parse_args(argv)
    global REPS
    REPS = a.reps
    cells = all_cells(dict(true_auc=[0.8], n=[200], icc=[0.0], scenario=a.scenario, p=a.p))
    tasks = []
    for c in cells:
        for m in a.models:
            st = stored_indep(a.rf_cells if m == "rf" else f"{a.run_dir}/cells", m, c["cell_id"])
            tasks.append((c, m, st, a.rf_check if m == "rf" else REPS))
    tasks.sort(key=lambda t: t[1] != "rf")                         # slow RF tasks first
    out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one_task)(*t) for t in tasks)
    df = pd.DataFrame(out).sort_values(["model", "scenario", "p"])
    path = Path(a.run_dir) / "test3_comparator80.csv"
    if path.exists():                       # keep rows for models not recomputed now
        old = pd.read_csv(path)
        df = pd.concat([old[~old.model.isin(a.models)], df], ignore_index=True)
    df.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(df[["model", "scenario", "p", "auc_indep80_mean", "auc_indep80_mcse",
              "regen_checked", "regen_max_abs_diff", "regen_ok"]].to_string(index=False))
    print(f"wrote {path}")
    if not df.regen_ok.all():
        print("WARNING: regeneration not verified for:",
              df.loc[~df.regen_ok, ["model", "scenario", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()

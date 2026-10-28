"""Aggregate per-cell replicate files into runs/<run_id>/results.csv.

One row per cell x model x split. Primary AUC = fold-mean (DECISIONS.md D4);
pooled-OOF columns are kept as a sensitivity analysis.

    python -m mie27.summarise --run-id v1
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import calibration_table

SPLIT_LABEL = {"row": "a_row", "unit": "b_unit", "patient": "c_patient"}
FALSE_ALARM_AT = 0.70


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src"]) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_replicates(run_dir: Path) -> pd.DataFrame:
    files = sorted((run_dir / "cells").glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"no cell files under {run_dir}/cells")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def summarise(reps: pd.DataFrame, run_id: str, ghash: str) -> pd.DataFrame:
    keys = ["cell_id", "scenario", "n", "icc", "p", "true_auc", "beta", "seed", "model"]
    out = []
    for key, g in reps.groupby(keys, sort=True):
        base = dict(zip(keys, key))
        for split, lab in SPLIT_LABEL.items():
            a = g[f"aucfm_{split}"].to_numpy()
            ap = g[f"auc_{split}"].to_numpy()
            opt = a - g["aucfm_patient"].to_numpy()
            optp = ap - g["auc_patient"].to_numpy()
            out.append(dict(
                **base, split=lab, n_reps=len(g),
                auc_mean=a.mean(), auc_q025=np.quantile(a, .025), auc_q975=np.quantile(a, .975),
                optimism_mean=opt.mean(),                       # (split) - (c); for a_row = (a)-(c)
                optimism_within_001=bool(abs(opt.mean()) < 0.01),
                auc_indep_mean=g["auc_indep"].mean() if "auc_indep" in g else np.nan,
                optimism_mcse=opt.std(ddof=1) / np.sqrt(len(opt)) if len(opt) > 1 else np.nan,
                optimism_q025=np.quantile(opt, .025), optimism_q975=np.quantile(opt, .975),
                false_alarm_070=(np.mean(a > FALSE_ALARM_AT) if base["true_auc"] == 0.5 else np.nan),
                auc_pooled_mean=ap.mean(), optimism_pooled_mean=optp.mean(),
                false_alarm_070_pooled=(np.mean(ap > FALSE_ALARM_AT) if base["true_auc"] == 0.5 else np.nan),
                skipped_folds_mean=g[f"skipfold_{split}"].mean(),
                k_mean=g["k_mean"].mean(), redraws_total=int(g["redraws"].sum()),
                runtime_s=g["runtime_s"].sum(),                 # whole cell, all splits, this model
                run_id=run_id,
                git_hash=(";".join(sorted(g["git_hash"].astype(str).unique()))
                          if "git_hash" in g else ghash)))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these models (e.g. drop rf under the deadline rule)")
    a = ap.parse_args(argv)
    run_dir = Path(a.runs_dir) / a.run_id
    reps = load_replicates(run_dir)
    if a.models:
        reps = reps[reps.model.isin(a.models)]
    reps.to_csv(run_dir / "replicates.csv.gz", index=False)
    res = summarise(reps, a.run_id, git_hash())
    res.to_csv(run_dir / "results.csv", index=False)
    cal = run_dir / "calibration.csv"
    if not cal.exists():
        calibration_table().to_csv(cal, index=False)
    print(f"{len(res)} rows -> {run_dir/'results.csv'}; models={sorted(res.model.unique())}; "
          f"reps/cell: min={res.n_reps.min()} max={res.n_reps.max()}")


if __name__ == "__main__":
    main()

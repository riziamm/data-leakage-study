"""Run the simulation grid.

Every cell has a seed derived from (base_seed, cell_id) only, so all models see
identical data and identical folds, and any subset/shard/machine reproduces the
same numbers. Results are written per (model, cell) so runs resume and RF shards
from Gadi can be merged with local LR/kNN output.

    python -m mie27.run --run-id v1 --models lr knn --reps 200 --jobs 8
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import subprocess
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .calibrate import beta_for
from .cv import N_REPEATS, make_folds, oof_auc
from .generate import SCENARIOS, simulate

BASE_SEED = 20271
N_INDEP = 2000      # patients (k = 1) in the independent test sample used for auc_indep


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """HEAD of the repo, with -dirty if src/ has uncommitted changes. Stored on every
    replicate row so extended / merged runs remain traceable to the code that made them."""
    root = Path(__file__).resolve().parents[2]
    try:
        h = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                    stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "src"],
                                stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
GRID = dict(scenario=list(SCENARIOS), n=[20, 30, 50, 100, 200],
            icc=[0.0, 0.3, 0.6, 0.9], p=[20, 180], true_auc=[0.50, 0.65, 0.80])


def cell_id(scenario, n, icc, p, true_auc) -> str:
    return f"{scenario}|n{n}|icc{icc:g}|p{p}|auc{true_auc:.2f}"


def cell_seed(cid: str, base_seed: int = BASE_SEED) -> int:
    h = hashlib.sha256(f"{base_seed}|{cid}".encode()).digest()
    return int.from_bytes(h[:8], "little") & (2**63 - 1)


def all_cells(filters: dict | None = None) -> list[dict]:
    g = {k: list(v) for k, v in GRID.items()}
    for k, v in (filters or {}).items():
        if v:
            g[k] = [x for x in g[k] if x in set(v)]
    cells = [dict(zip(g, vals)) for vals in itertools.product(*g.values())]
    for c in cells:
        c["cell_id"] = cell_id(**c)
        c["seed"] = cell_seed(c["cell_id"])
    return cells


def run_replicate(cell: dict, models, rep_ss: np.random.SeedSequence) -> dict[str, dict]:
    """One replicate for several models on the *same* data and folds.
    Data and folds depend only on rep_ss, never on which models are requested, so
    running models together or separately (e.g. RF on Gadi) gives identical numbers.
    runtime_s per model = shared cost (data + folds) + that model's own CV time."""
    if isinstance(models, str):
        models = [models]
    t0 = time.perf_counter()
    data_ss, cv_ss, indep_ss = rep_ss.spawn(3)
    rng = np.random.default_rng(data_ss)
    beta = beta_for(cell["true_auc"], cell["p"])
    d = simulate(cell["scenario"], cell["n"], cell["icc"], cell["p"], beta, rng)
    cv_rng = np.random.default_rng(cv_ss)
    cv_seeds = cv_rng.integers(0, 2**31 - 1, size=N_REPEATS)
    model_seed = int(cv_rng.integers(0, 2**31 - 1))
    grouped_unit = SCENARIOS[cell["scenario"]]["kind"] == "S1"
    splits = ("row", "unit", "patient") if grouped_unit else ("row", "patient")
    folds = {sp: [make_folds(sp, d["y"], d["patient"], d["unit"], int(s)) for s in cv_seeds]
             for sp in splits}
    # independent k = 1 test sample for the model's own out-of-sample AUC (test 3)
    d_ind = simulate("control", N_INDEP, cell["icc"], cell["p"], beta, np.random.default_rng(indep_ss))
    shared = time.perf_counter() - t0
    outs = {}
    for model in models:
        t1 = time.perf_counter()
        knn = None
        if model == "knn":
            from .fast import KNNScorer
            knn = KNNScorer(d["X"])
        out = {}
        for split in splits:
            res = [oof_auc(model, d["X"], d["y"], f, model_seed, knn) for f in folds[split]]
            out[f"auc_{split}"] = float(np.mean([r[0] for r in res]))           # pooled
            out[f"aucfm_{split}"] = float(np.nanmean([r[1] for r in res]))      # fold-mean
            out[f"skipfold_{split}"] = int(sum(r[2] for r in res))
        if not grouped_unit:                                                     # unit == row
            for pre in ("auc", "aucfm", "skipfold"):
                out[f"{pre}_unit"] = out[f"{pre}_row"]
        from .cv import make_model
        from sklearn.metrics import roc_auc_score
        full = make_model(model, model_seed).fit(d["X"], d["y"])
        out["auc_indep"] = float(roc_auc_score(d_ind["y"], full.predict_proba(d_ind["X"])[:, 1]))
        out.update(n_rows=len(d["y"]), k_mean=len(d["y"]) / cell["n"], redraws=d["redraws"],
                   runtime_s=shared + time.perf_counter() - t1)
        outs[model] = out
    return outs


def _cell_path(out_dir: Path, model: str, cell: dict, chunk=None) -> Path:
    h = hashlib.md5(cell["cell_id"].encode()).hexdigest()
    sfx = "" if chunk is None else f".c{chunk[0]}of{chunk[1]}"
    return out_dir / "cells" / model / (h + sfx + ".csv")


def _load_done(path: Path, rep_lo: int) -> pd.DataFrame | None:
    """Existing replicates, or None if absent/not a clean rep_lo..rep_lo+m-1 prefix."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if df["rep"].tolist() == list(range(rep_lo, rep_lo + len(df))) else None


def run_cell(cell: dict, models, reps: int, out_dir: Path, chunk=None) -> list[Path]:
    """Bring every requested model's file for this cell (or replicate chunk i of C)
    up to its replicate range. Replicate r always uses SeedSequence(cell_seed).spawn(reps)[r],
    so chunked, resumed or extended runs give exactly the same numbers as one run."""
    if isinstance(models, str):
        models = [models]
    rep_lo, rep_hi = 0, reps
    if chunk is not None:
        i, C = chunk
        rep_lo, rep_hi = i * reps // C, (i + 1) * reps // C
    done = {m: _load_done(_cell_path(out_dir, m, cell, chunk), rep_lo) for m in models}
    n_done = {m: rep_lo if done[m] is None else rep_lo + len(done[m]) for m in models}
    todo = [m for m in models if n_done[m] < rep_hi]
    if todo:
        start = min(n_done[m] for m in todo)
        rep_seqs = np.random.SeedSequence(cell["seed"]).spawn(reps)
        rows = {m: [] for m in todo}
        for r in range(start, rep_hi):
            need = [m for m in todo if n_done[m] <= r]
            for m, res in run_replicate(cell, need, rep_seqs[r]).items():
                res["rep"] = r
                rows[m].append(res)
        for m in todo:
            df = pd.DataFrame(rows[m])
            for k in ("cell_id", "scenario", "n", "icc", "p", "true_auc", "seed"):
                df[k] = cell[k]
            df["beta"] = beta_for(cell["true_auc"], cell["p"])
            df["model"] = m
            df["git_hash"] = code_hash()
            if done[m] is not None:
                df = pd.concat([done[m], df[done[m].columns]], ignore_index=True)
            path = _cell_path(out_dir, m, cell, chunk)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(path)
    return [_cell_path(out_dir, m, cell, chunk) for m in models]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models", nargs="+", default=["lr", "knn"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1,
                    help="split each cell's replicates into this many tasks (use ~10 for RF)")
    ap.add_argument("--shard", default="0/1", help="i/N: run every N-th cell starting at i")
    for k in GRID:
        ap.add_argument(f"--{k.replace('_', '-')}", nargs="*",
                        type=str if k == "scenario" else (int if k in ("n", "p") else float))
    a = ap.parse_args(argv)
    filters = {k: getattr(a, k) for k in GRID}
    cells = all_cells(filters)
    i, N = map(int, a.shard.split("/"))
    cells = cells[i::N]
    out_dir = Path(a.runs_dir) / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # models share data + folds within a cell; cells are ordered so the slowest
    # (largest n x k x p) start first, which keeps parallel wall time close to optimal
    cells.sort(key=lambda c: -c["n"] * {"S2_k8": 8, "S3": 5, "S2_k4": 4, "S1_r2_d0": 4,
                                         "S1_r2_d0.2": 4}.get(c["scenario"], 2) * c["p"])
    print(f"{len(cells)} cells x models {a.models}, {a.reps} reps each, jobs={a.jobs}", flush=True)
    t0 = time.time()
    tasks = [(c, None) for c in cells] if a.chunks == 1 else \
            [(c, (i, a.chunks)) for c in cells for i in range(a.chunks)]
    print(f"{len(tasks)} tasks", flush=True)
    Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_cell)(c, a.models, a.reps, out_dir, ch)
                                       for c, ch in tasks)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

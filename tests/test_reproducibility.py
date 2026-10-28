"""Seeding: per-cell results do not depend on run length, model grouping or order."""
import numpy as np
import pandas as pd

from mie27.run import _cell_path, all_cells, run_cell

CELL = next(c for c in all_cells() if c["cell_id"] == "S3|n20|icc0.6|p20|auc0.65")
COLS = ["aucfm_row", "aucfm_unit", "aucfm_patient", "auc_row", "auc_patient", "k_mean"]


def _read(d, m):
    return pd.read_csv(_cell_path(d, m, CELL))[COLS].to_numpy()


def test_extending_replicates_equals_running_at_once(tmp_path):
    run_cell(CELL, ["lr", "knn"], 3, tmp_path / "a")
    run_cell(CELL, ["lr", "knn"], 5, tmp_path / "a")       # extend 3 -> 5
    run_cell(CELL, ["lr", "knn"], 5, tmp_path / "b")       # 5 at once
    for m in ("lr", "knn"):
        np.testing.assert_array_equal(_read(tmp_path / "a", m), _read(tmp_path / "b", m))


def test_models_run_separately_see_same_data(tmp_path):
    run_cell(CELL, ["lr", "knn"], 3, tmp_path / "joint")
    run_cell(CELL, ["knn"], 3, tmp_path / "solo")
    np.testing.assert_array_equal(_read(tmp_path / "joint", "knn"), _read(tmp_path / "solo", "knn"))

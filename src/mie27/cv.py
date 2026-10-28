"""Splits (5-fold x 3 repeats) and models (default settings, no tuning)."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .fast import stratified_group_kfold

N_SPLITS, N_REPEATS = 5, 3
SPLITS = ("row", "unit", "patient")   # (a), (b), (c)
MODELS = ("lr", "knn", "rf")


class LeakageError(AssertionError):
    pass


def make_model(name: str, seed: int):
    if name == "lr":
        # L2, C = 1 are sklearn defaults; max_iter raised only so lbfgs converges at p = 180
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(name)


def assert_grouped(folds: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray) -> None:
    """Each group appears in exactly one test fold and never in train and test at once."""
    seen = np.zeros(groups.max() + 1, dtype=int)
    for tr, te in folds:
        g_te = np.unique(groups[te])
        if np.intersect1d(np.unique(groups[tr]), g_te).size:
            raise LeakageError("a group is in both train and test of one fold")
        seen[g_te] += 1
    present = np.unique(groups)
    if not np.all(seen[present] == 1):
        raise LeakageError("a group is placed in more than one test fold")


def make_folds(split: str, y: np.ndarray, patient: np.ndarray, unit: np.ndarray,
               seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    X0 = np.zeros((len(y), 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # small-class warnings at n = 20
        if split == "row":
            folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X0, y))
        elif split in ("unit", "patient"):
            groups = unit if split == "unit" else patient
            # exact replica of StratifiedGroupKFold(shuffle=True), vectorised (fast.py)
            folds = stratified_group_kfold(y, groups, N_SPLITS, seed)
            assert_grouped(folds, groups)
        else:
            raise ValueError(split)
    return folds


def oof_auc(model_name: str, X, y, folds, model_seed: int, knn=None) -> tuple[float, float, int]:
    """One repeat -> (pooled OOF AUC, mean of per-fold AUCs, n single-class test folds).
    See DECISIONS.md D4 for why both are kept. LR/kNN use the exact fast paths in
    fast.py (equivalence tested in tests/test_fast_equivalence.py); RF uses sklearn."""
    from .fast import KNNScorer, auc, lr_scores
    oof = np.empty(len(y))
    fold_aucs, skipped = [], 0
    for tr, te in folds:
        if np.unique(y[tr]).size < 2:
            oof[te] = y[tr].mean()
        elif model_name == "lr":
            oof[te] = lr_scores(X[tr], y[tr], X[te])
        elif model_name == "knn":
            oof[te] = (knn or KNNScorer(X)).scores(tr, te, y)
        else:
            m = make_model(model_name, model_seed).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        if np.unique(y[te]).size == 2:
            fold_aucs.append(auc(y[te], oof[te]))
        else:
            skipped += 1
    fm = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    return auc(y, oof), fm, skipped

"""Exact fast paths for the inner CV loop (no change to any estimator).

* auc():       Mann-Whitney AUC with mid-ranks == sklearn.metrics.roc_auc_score
               (identical, ties included), without per-call input validation.
* KNNScorer:   kNN (k = 5, Euclidean, uniform weights) predict_proba[:, 1] from one
               pairwise-distance matrix per replicate, sliced per fold. Equivalent to
               sklearn KNeighborsClassifier(n_neighbors=5) except for exact distance
               ties, which have probability 0 with continuous features.
* lr_scores(): StandardScaler + LogisticRegression(C=1) done inline; still the
               sklearn lbfgs estimator, only the Pipeline wrapper is removed.

Equivalence to the sklearn reference implementations is asserted in
tests/test_fast_equivalence.py.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

K_NEIGHBORS = 5


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = y == 1
    n1 = int(pos.sum())
    n0 = y.size - n1
    r = rankdata(s)                       # average ranks for ties
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class KNNScorer:
    def __init__(self, X: np.ndarray):
        sq = np.einsum("ij,ij->i", X, X)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        self.D = D

    def scores(self, tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(K_NEIGHBORS, tr.size)
        Dsub = self.D[np.ix_(te, tr)]
        if k < tr.size:
            nn = np.argpartition(Dsub, k - 1, axis=1)[:, :k]
        else:
            nn = np.broadcast_to(np.arange(tr.size), (te.size, tr.size))
        return y[tr][nn].mean(axis=1)


def lr_scores(Xtr, ytr, Xte) -> np.ndarray:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0                     # StandardScaler's zero-variance rule
    m = LogisticRegression(C=1.0, max_iter=1000).fit((Xtr - mu) / sd, ytr)
    return m.predict_proba((Xte - mu) / sd)[:, 1]


def _isclose(a: float, b: float, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """np.isclose for two scalars (b may be inf)."""
    if a == b:
        return True
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


def stratified_group_kfold(y: np.ndarray, groups: np.ndarray, n_splits: int,
                           seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold-for-fold replica of sklearn StratifiedGroupKFold(n_splits, shuffle=True,
    random_state=seed).split(X, y, groups) (sklearn 1.9.1 algorithm: greedy assignment
    of groups, sorted by class-count std, to the fold minimising mean per-class std,
    ties -> fewest samples). The per-group candidate evaluation is vectorised over the
    n_splits folds with the same arithmetic; identity with sklearn is asserted in
    tests/test_fast_equivalence.py across all scenarios and n."""
    y = np.asarray(y)
    _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
    if np.all(n_splits > y_cnt):
        raise ValueError("n_splits greater than the number of members in each class")
    n_classes = len(y_cnt)
    _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
    n_groups = len(groups_cnt)
    if n_splits > n_groups:
        raise ValueError("n_splits greater than the number of groups")
    ycpg = np.zeros((n_groups, n_classes))
    np.add.at(ycpg, (groups_inv, y_inv), 1)
    rng = np.random.RandomState(seed)
    perm = np.arange(n_groups)
    rng.shuffle(perm)
    ycpg = ycpg[perm]
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size)
    groups_inv = inv_perm[groups_inv]
    order = np.argsort(-np.std(ycpg, axis=1), kind="stable")

    ycpf = np.zeros((n_splits, n_classes))
    eye = np.eye(n_splits)[:, :, None]
    ycnt = y_cnt.reshape(1, 1, -1)
    fold_of_group = np.empty(n_groups, dtype=int)
    for g in order:
        gc = ycpg[g]
        cand = (ycpf[None, :, :] + eye * gc[None, None, :]) / ycnt   # (cand, fold, class)
        evals = np.std(cand, axis=1).mean(axis=1)
        sizes = ycpf.sum(axis=1)
        best, min_eval, min_s = None, np.inf, np.inf
        for i in range(n_splits):
            fe, s = float(evals[i]), float(sizes[i])
            if fe < min_eval or (_isclose(fe, min_eval) and s < min_s):
                best, min_eval, min_s = i, fe, s
        ycpf[best] += gc
        fold_of_group[g] = best
    row_fold = fold_of_group[groups_inv]
    idx = np.arange(len(y))
    return [(idx[row_fold != i], idx[row_fold == i]) for i in range(n_splits)]

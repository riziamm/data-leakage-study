"""Beta calibration.

"True AUC" = AUC of the oracle score w'x on an independent k = 1 test set
(see DECISIONS.md D2). Because x | y ~ N(beta*y*w, I_p) and ||w|| = 1,
w'x | y ~ N(beta*y, 1) and AUC = Phi(beta / sqrt(2)), so beta = sqrt(2)*Phi^-1(AUC)
for every p. We still calibrate per p by Monte Carlo root-finding and store both,
so the table documents that p does not matter for the oracle.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .generate import simulate

TARGET_AUCS = (0.50, 0.65, 0.80)
P_VALUES = (20, 180)


def beta_analytic(auc: float) -> float:
    return float(np.sqrt(2.0) * NormalDist().inv_cdf(auc)) if auc != 0.5 else 0.0


def oracle_auc(beta: float, p: int, n_test: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    d = simulate("control", n_test, 0.0, p, beta, rng)
    w = np.zeros(p)
    w[:5] = 1 / np.sqrt(5)
    return float(roc_auc_score(d["y"], d["X"] @ w))


def calibrate_mc(target: float, p: int, n_test: int = 100_000, seed: int = 2027,
                 tol: float = 1e-4) -> float:
    """Bisection on beta using a common-random-numbers oracle AUC."""
    if target == 0.5:
        return 0.0
    lo, hi = 0.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if oracle_auc(mid, p, n_test, seed) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def calibration_table(n_test: int = 100_000, seed: int = 2027) -> pd.DataFrame:
    rows = []
    for p in P_VALUES:
        for a in TARGET_AUCS:
            b_an = beta_analytic(a)
            b_mc = calibrate_mc(a, p, n_test, seed)
            # verify on fresh, independent seed
            achieved = oracle_auc(b_an, p, n_test, seed + 1)
            rows.append(dict(p=p, target_auc=a, beta=b_an, beta_mc=b_mc,
                             achieved_auc_check=achieved, n_test=n_test))
    return pd.DataFrame(rows)


def beta_for(auc: float, p: int) -> float:
    """Beta used in the simulation (analytic; MC-verified in the stored table)."""
    return beta_analytic(auc)

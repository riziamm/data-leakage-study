"""The fast inner-loop paths must reproduce the sklearn reference estimators."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mie27.cv import make_folds
from mie27.fast import KNNScorer, auc, lr_scores
from mie27.generate import simulate


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = rng.integers(4, 400)
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        s = rng.integers(0, 6, n) / 5 if rng.random() < .5 else rng.standard_normal(n)
        assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


@pytest.mark.parametrize("scenario,n,p", [("S2_k8", 50, 180), ("S3", 30, 20), ("S1_r2_d0.2", 20, 180)])
def test_knn_matches_sklearn(scenario, n, p):
    d = simulate(scenario, n, 0.6, p, 0.8, np.random.default_rng(1))
    sc = KNNScorer(d["X"])
    for split in ("row", "patient"):
        for tr, te in make_folds(split, d["y"], d["patient"], d["unit"], 3):
            ref = KNeighborsClassifier(n_neighbors=5).fit(d["X"][tr], d["y"][tr]).predict_proba(d["X"][te])[:, 1]
            np.testing.assert_allclose(sc.scores(tr, te, d["y"]), ref, atol=1e-12)


@pytest.mark.parametrize("scenario,n,p", [("S2_k8", 50, 180), ("S3", 30, 20), ("control", 20, 180)])
def test_lr_matches_sklearn_pipeline(scenario, n, p):
    d = simulate(scenario, n, 0.6, p, 0.8, np.random.default_rng(2))
    for tr, te in make_folds("patient", d["y"], d["patient"], d["unit"], 5):
        ref = (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
               .fit(d["X"][tr], d["y"][tr]).predict_proba(d["X"][te])[:, 1])
        np.testing.assert_allclose(lr_scores(d["X"][tr], d["y"][tr], d["X"][te]), ref, atol=1e-8)


@pytest.mark.parametrize("scenario", ["control", "S1_r1_d0.2", "S1_r2_d0.2", "S2_k8", "S3"])
def test_stratified_group_kfold_identical_to_sklearn(scenario):
    from sklearn.model_selection import StratifiedGroupKFold
    from mie27.fast import stratified_group_kfold
    rng = np.random.default_rng(11)
    for n in (20, 30, 50, 100, 200):
        for rep in range(8):
            d = simulate(scenario, n, 0.3, 20, 0.5, rng)
            for groups in (d["patient"], d["unit"]):
                seed = int(rng.integers(0, 2**31 - 1))
                ref = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                           .split(np.zeros((len(d["y"]), 1)), d["y"], groups))
                got = stratified_group_kfold(d["y"], groups, 5, seed)
                for (rtr, rte), (gtr, gte) in zip(ref, got):
                    np.testing.assert_array_equal(rte, gte)
                    np.testing.assert_array_equal(rtr, gtr)

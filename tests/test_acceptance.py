"""Acceptance tests 0-5 (0 = replication requirement, added; see DECISIONS.md D7)."""
import numpy as np
import pandas as pd
import pytest

from mie27.calibrate import beta_for, oracle_auc
from mie27.cv import LeakageError, assert_grouped, make_folds
from mie27.generate import SCENARIOS, simulate

from conftest import REQUIRED_REPS, TOL_OPT


def _report(bad: pd.DataFrame, cols) -> str:
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        return f"\n{len(bad)} failing rows:\n" + bad[cols].to_string(index=False)


# ---- 0. replication ---------------------------------------------------------
def test_0_full_replication(results):
    """The spec's 200 replicates per cell; tests 1-3 are only meaningful at that size."""
    assert results.n_reps.min() >= REQUIRED_REPS, (
        f"min replicates/cell = {results.n_reps.min()} < {REQUIRED_REPS}")


# ---- 1. ICC = 0 -> no optimism ------------------------------------------------
# Gate (DECISIONS.md D7, accepted 28 Oct): per cell |z| < 3 where z = mean / MC SE,
# and at most 1% of cells may fail. |optimism| < 0.01 is reported, not gated.
Z_MAX, FRAC_MAX = 3.0, 0.01


def _z_gate(d: pd.DataFrame, cols) -> None:
    d = d.copy()
    d["z"] = d.optimism_mean / d.optimism_mcse
    bad = d[d.z.abs() >= Z_MAX]
    frac = len(bad) / len(d)
    n_abs = int((d.optimism_mean.abs() >= TOL_OPT).sum())
    msg = (f"\n{len(bad)}/{len(d)} cells with |z| >= {Z_MAX} ({frac:.1%}; limit {FRAC_MAX:.0%}); "
           f"{n_abs} cells with |optimism| >= {TOL_OPT} (reported only)")
    assert frac <= FRAC_MAX, msg + _report(bad, cols + ["optimism_mean", "optimism_mcse", "z", "n_reps"])


@pytest.mark.parametrize("model", ["lr", "knn", "rf"])
def test_1_icc0_no_optimism(results, model):
    d = results[(results.icc == 0) & (results.split == "a_row") & (results.model == model)]
    if d.empty:
        pytest.skip(f"no {model} results in this run")
    _z_gate(d, ["scenario", "n", "p", "true_auc"])


# ---- 2. control k = 1 -> no optimism (same gate as test 1) ---------------------
@pytest.mark.parametrize("model", ["lr", "knn", "rf"])
def test_2_control_no_optimism(results, model):
    d = results[(results.scenario == "control") & (results.split == "a_row")
                & (results.model == model)]
    if d.empty:
        pytest.skip(f"no {model} results in this run")
    _z_gate(d, ["n", "icc", "p", "true_auc"])


# ---- 3. AUC 0.80, n = 200, ICC = 0 -> every split within 0.02 of the model's own
#         independent-test AUC when trained on 80% of patients (the CV training size).
#         Comparator from scripts/test3_comparator.py (approved 29 Oct; the earlier
#         100%-trained comparator was biased high by the learning curve, D3b).
def test_3_splits_match_independent_auc(results):
    from conftest import run_dir
    d = results[(results.true_auc == 0.8) & (results.n == 200) & (results.icc == 0)].copy()
    assert not d.empty
    path = run_dir() / "test3_comparator80.csv"
    assert path.exists(), f"missing {path}: run scripts/test3_comparator.py first"
    comp = pd.read_csv(path)
    missing = sorted(set(d.model) - set(comp.model))
    assert not missing, f"comparator lacks models {missing}"
    unverified = comp[comp.model.isin(d.model) & ~comp.regen_ok.astype(bool)]
    assert unverified.empty, "replicate regeneration not verified:\n" + unverified.to_string()
    d = d.merge(comp[["cell_id", "model", "auc_indep80_mean", "auc_indep80_mcse"]],
                on=["cell_id", "model"], how="left")
    assert d.auc_indep80_mean.notna().all()
    d["gap"] = d.auc_mean - d.auc_indep80_mean
    bad = d[d.gap.abs() > 0.02]
    assert bad.empty, _report(bad, ["scenario", "p", "model", "split", "auc_mean",
                                    "auc_indep80_mean", "auc_indep_mean", "gap", "n_reps"])


# ---- 3b. variance components sum to 1 on the full grid ----------------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("icc", [0.0, 0.3, 0.6, 0.9])
def test_3b_variance_components_sum_to_one(scenario, icc):
    from mie27.generate import variance_components
    v = variance_components(scenario, icc)
    assert all(x >= 0 for x in v.values()), v
    assert abs(sum(v.values()) - 1.0) < 1e-12, v
    simulate(scenario, 20, icc, 20, 0.5, np.random.default_rng(0))   # in-code assertion


# ---- 4. patient-grouped split never puts a patient in two folds -------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("n", [20, 200])
def test_4_patient_split_disjoint(scenario, n):
    rng = np.random.default_rng(4)
    for rep in range(10):
        d = simulate(scenario, n, 0.6, 20, 0.5, rng)
        for seed in range(3):
            folds = make_folds("patient", d["y"], d["patient"], d["unit"], seed)
            test_patients = [set(d["patient"][te]) for _, te in folds]
            # every patient in exactly one test fold, never in train and test together
            assert sum(len(s) for s in test_patients) == len(set().union(*test_patients)) == n
            for tr, te in folds:
                assert not set(d["patient"][tr]) & set(d["patient"][te])
            if SCENARIOS[scenario]["kind"] == "S1":
                uf = make_folds("unit", d["y"], d["patient"], d["unit"], seed)
                for tr, te in uf:
                    assert not set(d["unit"][tr]) & set(d["unit"][te])


def test_4b_guard_fires_on_leaky_folds():
    """The runtime guard inside make_folds must reject a row split of clustered data."""
    rng = np.random.default_rng(0)
    d = simulate("S2_k4", 30, 0.6, 20, 0.5, rng)
    leaky = make_folds("row", d["y"], d["patient"], d["unit"], 0)
    with pytest.raises(LeakageError):
        assert_grouped(leaky, d["patient"])


# ---- 5. beta calibration reproduces target AUC within 0.01 ------------------
@pytest.mark.parametrize("p", [20, 180])
@pytest.mark.parametrize("target", [0.50, 0.65, 0.80])
def test_5_calibration(p, target):
    got = oracle_auc(beta_for(target, p), p, n_test=100_000, seed=777 + p)
    assert abs(got - target) < 0.01, f"p={p} target={target} achieved={got:.4f}"

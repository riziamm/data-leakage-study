"""Synthetic generative model. No real patient data is used or stored anywhere.

x_row = sqrt(ICC_p)*u_patient [+ sqrt(s2_eye)*v_eye] + sqrt(s2_res)*e_row + beta*y_row*w

Every feature has total variance 1 in every scenario, so x | y ~ N(beta*y*w, I_p)
marginally and the oracle score w'x gives AUC = Phi(beta / sqrt(2)) whatever the
ICC, scenario or p.

S1 nesting (see DECISIONS.md D1): ICC_e is the eye's share of the *within-patient*
variance, s2_eye = ICC_e * (1 - ICC_p), s2_res = (1 - ICC_p) * (1 - ICC_e).
Corr(two eyes, same patient) = ICC_p; Corr(two repeats, same eye) = ICC_p + s2_eye.
"""
from __future__ import annotations

import numpy as np

N_SIGNAL = 5

SCENARIOS: dict[str, dict] = {
    "control": dict(kind="control"),
    "S1_r1_d0": dict(kind="S1", r=1, d=0.0),
    "S1_r1_d0.2": dict(kind="S1", r=1, d=0.2),
    "S1_r2_d0": dict(kind="S1", r=2, d=0.0),
    "S1_r2_d0.2": dict(kind="S1", r=2, d=0.2),
    "S2_k4": dict(kind="S2", k=4),
    "S2_k8": dict(kind="S2", k=8),
    "S3": dict(kind="S3", nb_mean=4.0, nb_disp=1.0, cap=50),
}


def signal_vector(p: int) -> np.ndarray:
    """Fixed sparse unit vector on the first 5 features."""
    w = np.zeros(p)
    w[:N_SIGNAL] = 1.0 / np.sqrt(N_SIGNAL)
    return w


def rows_per_patient(scenario: str, n: int, rng: np.random.Generator) -> np.ndarray:
    s = SCENARIOS[scenario]
    if s["kind"] == "control":
        return np.ones(n, dtype=int)
    if s["kind"] == "S1":
        return np.full(n, 2 * s["r"], dtype=int)
    if s["kind"] == "S2":
        return np.full(n, s["k"], dtype=int)
    # S3: 1 + NegBin(mean m, dispersion size r); numpy NB(n=r, p) has mean r(1-p)/p
    r, m = s["nb_disp"], s["nb_mean"]
    k = 1 + rng.negative_binomial(r, r / (r + m), size=n)
    return np.minimum(k, s["cap"]).astype(int)


def simulate(scenario: str, n: int, icc: float, p: int, beta: float,
             rng: np.random.Generator, icc_e: float | None = None,
             min_per_class: int = 2) -> dict:
    """One replicate. Redraws patient labels until each class has >= min_per_class
    patients (degenerate draws make CV AUC undefined); the redraw count is returned."""
    s = SCENARIOS[scenario]
    w = signal_vector(p)
    redraws = 0
    while True:
        y_pat = rng.binomial(1, 0.5, size=n)
        if min(y_pat.sum(), n - y_pat.sum()) >= min_per_class:
            break
        redraws += 1

    if s["kind"] == "S1":
        r, d = s["r"], s["d"]
        icc_e = icc / 2.0 if icc_e is None else icc_e
        n_eyes = 2 * n
        eye_patient = np.repeat(np.arange(n), 2)
        flip = rng.random(n_eyes) < d
        y_eye = y_pat[eye_patient] ^ flip.astype(int)
        row_eye = np.repeat(np.arange(n_eyes), r)
        patient = eye_patient[row_eye]
        y = y_eye[row_eye]
        s2_eye = icc_e * (1.0 - icc)
        s2_res = (1.0 - icc) * (1.0 - icc_e)
        assert abs(icc + s2_eye + s2_res - 1.0) < 1e-12, (icc, s2_eye, s2_res)
        N = len(y)
        u = rng.standard_normal((n, p))
        v = rng.standard_normal((n_eyes, p))
        e = rng.standard_normal((N, p))
        X = (np.sqrt(icc) * u[patient] + np.sqrt(s2_eye) * v[row_eye]
             + np.sqrt(s2_res) * e + beta * y[:, None] * w)
        unit = row_eye
    else:
        assert abs(icc + (1.0 - icc) - 1.0) < 1e-12
        k = rows_per_patient(scenario, n, rng)
        patient = np.repeat(np.arange(n), k)
        y = y_pat[patient]
        N = len(y)
        u = rng.standard_normal((n, p))
        e = rng.standard_normal((N, p))
        X = np.sqrt(icc) * u[patient] + np.sqrt(1.0 - icc) * e + beta * y[:, None] * w
        unit = np.arange(N)

    return dict(X=X, y=y.astype(int), patient=patient, unit=unit,
                y_patient=y_pat, redraws=redraws)


def variance_components(scenario: str, icc: float, icc_e: float | None = None) -> dict:
    """Variance shares of one feature; must sum to 1 (asserted over the grid in tests)."""
    if SCENARIOS[scenario]["kind"] == "S1":
        icc_e = icc / 2.0 if icc_e is None else icc_e
        return dict(patient=icc, eye=icc_e * (1 - icc), residual=(1 - icc) * (1 - icc_e))
    return dict(patient=icc, eye=0.0, residual=1 - icc)

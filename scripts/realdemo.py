"""Real-data demonstration of the v1 optimism calculator (run_id realdemo-v1).

Question, strictly: does calculator.py predict the row-vs-patient CV optimism observed
on a public multi-row dataset? No new modelling claims.

Data: PhysioNet/CinC 2019 Challenge, training set A (one PSV file per patient,
hourly rows). See DATA.md for URL and licence (CC BY 4.0).

Design (as specified):
  patient label  = any SepsisLabel == 1 in the record
  features       = the 20 vitals/labs with the lowest missingness (p = 20),
                   median-imputed and standardised on the training fold
  rows/patient   = first k hourly rows, k = 4 and (if enough patients) k = 8
  subsamples     = n in {50, 100, 200} patients, class-balanced, 100 per n, seeded
  pipeline       = identical to v1: 5-fold x 3 repeats, splits (a) row / (c) patient,
                   fold-mean AUC, LR / 5-NN / RF with v1 settings (mie27.cv.make_folds)
  ICC            = one-way random-effects ANOVA per feature on the full eligible cohort
                   at the same k (median and IQR and mean reported)
  calculator     = optimism(n, k, ICC_mean, p=20, model, true_auc=observed patient-split AUC)

    python scripts/realdemo.py --data /path/to/training_setA --jobs 32
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import skew  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mie27.cv import N_REPEATS, make_folds  # noqa: E402  (v1 fold scheme, unchanged)

RUN_ID = "realdemo-v1"
BASE_SEED = 20271
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
P = 20
NS = (50, 100, 200)
KS = (4, 8)
N_SUB = 100
MODELS = ("lr", "knn", "rf")
MIN_PER_CLASS_FOR_K = 100      # k = 8 only if >= 100 eligible patients per class


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src", "scripts"]) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ data
def load(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.psv")))
    if not files:
        sys.exit(f"no .psv files in {data_dir}")
    parts = []
    for f in files:
        d = pd.read_csv(f, sep="|")
        d["pid"] = os.path.basename(f).split(".")[0]
        d["t"] = np.arange(len(d))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def cohort(df: pd.DataFrame, k: int):
    """First k rows of every patient with >= k rows; patient label = any SepsisLabel."""
    n_rows = df.groupby("pid").size()
    lab = df.groupby("pid").SepsisLabel.max()
    keep = n_rows.index[n_rows >= k]
    d = df[df.pid.isin(keep) & (df.t < k)].copy()
    d["y_pat"] = d.pid.map(lab).astype(int)
    info = dict(k=k, n_patients_total=int(len(n_rows)), n_eligible=int(len(keep)),
                frac_excluded=float(1 - len(keep) / len(n_rows)),
                n_eligible_sepsis=int(lab.loc[keep].sum()),
                n_eligible_nonsepsis=int((1 - lab.loc[keep]).sum()))
    return d, info


def select_features(d4: pd.DataFrame):
    miss = d4[VITALS + LABS].isna().mean().sort_values()
    feats = list(miss.index[:P])
    tab = pd.DataFrame(dict(missingness=miss))
    tab["selected"] = tab.index.isin(feats)
    tab["skewness_observed"] = [skew(d4[c].dropna()) if d4[c].notna().sum() > 2 else np.nan for c in tab.index]
    return feats, tab


# ------------------------------------------------------------------ ICC
def icc_balanced(x: np.ndarray, k: int) -> float:
    """One-way random-effects ICC(1), balanced design; x shape (n_patients, k)."""
    n = x.shape[0]
    gm = x.mean()
    pm = x.mean(1)
    msb = k * ((pm - gm) ** 2).sum() / (n - 1)
    msw = ((x - pm[:, None]) ** 2).sum() / (n * (k - 1))
    if msb + (k - 1) * msw == 0:
        return np.nan
    return float((msb - msw) / (msb + (k - 1) * msw))


def icc_unbalanced(values: pd.Series, groups: pd.Series) -> float:
    """One-way ICC(1) on observed values only (unbalanced: k0 correction)."""
    v = pd.DataFrame(dict(x=values, g=groups)).dropna()
    cnt = v.groupby("g").x.count()
    v = v[v.g.isin(cnt.index[cnt >= 2])]
    if v.g.nunique() < 2:
        return np.nan
    n_i = v.groupby("g").x.count(); N = len(v); a = len(n_i)
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    gm = v.x.mean(); pm = v.groupby("g").x.mean()
    msb = (n_i * (pm - gm) ** 2).sum() / (a - 1)
    msw = ((v.x - v.g.map(pm)) ** 2).sum() / (N - a)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def icc_table(d: pd.DataFrame, feats, k: int) -> pd.DataFrame:
    rows = []
    med = d[feats].median()
    for f in feats:
        imp = d[f].fillna(med[f]).to_numpy().reshape(-1, k)        # rows sorted by pid, t
        rows.append(dict(feature=f, k=k, icc_imputed=icc_balanced(imp, k),
                         icc_observed_only=icc_unbalanced(d[f], d.pid),
                         frac_patients_all_missing=float((d.groupby("pid")[f].count() == 0).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pipeline (v1-identical CV)
def make_model(name: str, seed: int):
    # imputer + scaler fitted on the training fold only; classifier settings as in v1
    if name == "lr":
        clf = LogisticRegression(C=1.0, max_iter=1000)
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    else:
        raise ValueError(name)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def foldmean_auc(model, X, y, folds, seed) -> float:
    aucs = []
    for tr, te in folds:
        if np.unique(y[tr]).size < 2:
            pred = np.full(len(te), y[tr].mean())
        else:
            pred = make_model(model, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if np.unique(y[te]).size == 2:
            aucs.append(roc_auc_score(y[te], pred))
    return float(np.mean(aucs)) if aucs else np.nan


def draw_subsample(d: pd.DataFrame, feats, n: int, seed: int):
    """Class-balanced patient subsample (n/2 per class), drawn in the parent process."""
    rng = np.random.default_rng(seed)
    ypat = d.groupby("pid").y_pat.first()
    chosen = np.concatenate([rng.choice(ypat.index[ypat == c].to_numpy(), n // 2, replace=False) for c in (0, 1)])
    sub = d[d.pid.isin(chosen)]
    return (sub[feats].to_numpy(float), sub.y_pat.to_numpy(), pd.factorize(sub.pid)[0],
            int(rng.integers(0, 2**31 - 1)))


def one_subsample(X, y, patient, k, n, s, seed) -> list[dict]:
    rng = np.random.default_rng(seed)
    cv_seeds = rng.integers(0, 2**31 - 1, size=N_REPEATS)
    model_seed = int(rng.integers(0, 2**31 - 1))
    folds = {sp: [make_folds(sp, y, patient, np.arange(len(y)), int(cs)) for cs in cv_seeds]
             for sp in ("row", "patient")}
    out = []
    for model in MODELS:
        t0 = time.perf_counter()
        a = np.mean([foldmean_auc(model, X, y, f, model_seed) for f in folds["row"]])
        c = np.mean([foldmean_auc(model, X, y, f, model_seed) for f in folds["patient"]])
        out.append(dict(k=k, n=n, sub=s, model=model, auc_row=a, auc_patient=c,
                        optimism=a - c, runtime_s=time.perf_counter() - t0))
    return out


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with p*.psv files (training_setA)")
    ap.add_argument("--calculator", default="runs/v1/calculator.py")
    ap.add_argument("--out", default=f"runs/{RUN_ID}")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--n-sub", type=int, default=N_SUB)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ghash = git_hash()
    sys.path.insert(0, str(Path(a.calculator).parent))
    from calculator import optimism as calc_optimism, FIT_ERROR  # noqa: F401

    print(f"[{RUN_ID}] git {ghash}; loading {a.data} ...", flush=True)
    df = load(a.data)
    d4, info4 = cohort(df, 4)
    feats, ftab = select_features(d4)
    ftab.to_csv(out / "features_missingness_skewness.csv")
    print("features:", feats, flush=True)

    cohorts, infos = {}, {}
    for k in KS:
        d, info = cohort(df, k)
        if min(info["n_eligible_sepsis"], info["n_eligible_nonsepsis"]) < MIN_PER_CLASS_FOR_K:
            print(f"k = {k}: only {info} eligible -> skipped", flush=True)
            continue
        d = d.sort_values(["pid", "t"])
        cohorts[k], infos[k] = d, info

    # ICC per feature, full eligible cohort at each k
    icc = pd.concat([icc_table(cohorts[k], feats, k) for k in cohorts], ignore_index=True)
    icc.to_csv(out / "icc_per_feature.csv", index=False)
    icc_sum = icc.groupby("k").agg(icc_mean=("icc_imputed", "mean"), icc_median=("icc_imputed", "median"),
                                   icc_q25=("icc_imputed", lambda s: s.quantile(.25)),
                                   icc_q75=("icc_imputed", lambda s: s.quantile(.75)),
                                   icc_obs_mean=("icc_observed_only", "mean"),
                                   icc_obs_median=("icc_observed_only", "median"))
    print(icc_sum.to_string(), flush=True)

    # subsampled CV runs
    tasks = []
    for k, d in cohorts.items():
        for n in NS:
            for s in range(a.n_sub):
                seed = int(np.random.SeedSequence([BASE_SEED, k, n, s]).generate_state(1)[0])
                X, y, patient, cv_seed = draw_subsample(d, feats, n, seed)
                tasks.append((X, y, patient, k, n, s, cv_seed))
    tasks.sort(key=lambda t: -t[3] * t[4])
    print(f"{len(tasks)} subsamples x {len(MODELS)} classifiers on {a.jobs} workers", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one_subsample)(*t) for t in tasks)
    rep = pd.DataFrame([r for rr in res for r in rr])
    rep["run_id"], rep["git_hash"] = RUN_ID, ghash
    rep.to_csv(out / "subsample_results.csv.gz", index=False)

    # table: observed vs predicted
    rows = []
    for (k, n, m), g in rep.groupby(["k", "n", "model"]):
        obs, se = g.optimism.mean(), g.optimism.std(ddof=1) / np.sqrt(len(g))
        c_auc = g.auc_patient.mean()
        ta = float(np.clip(c_auc, 0.5, 0.8))
        note = [] if ta == c_auc else [f"true_auc clipped from {c_auc:.3f}"]
        preds = {}
        for lab, val in (("icc_mean", icc_sum.loc[k, "icc_mean"]), ("icc_median", icc_sum.loc[k, "icc_median"]),
                         ("icc_obs_mean", icc_sum.loc[k, "icc_obs_mean"])):
            try:
                preds[lab] = calc_optimism(n, k, float(val), P, m, true_auc=ta)
            except ValueError as e:
                preds[lab] = np.nan; note.append(f"{lab}: {e}")
        rows.append(dict(run_id=RUN_ID, git_hash=ghash, k=k, n=n, model=m, n_subsamples=len(g),
                         auc_row_mean=g.auc_row.mean(), auc_patient_mean=c_auc,
                         auc_patient_se=g.auc_patient.std(ddof=1) / np.sqrt(len(g)),
                         optimism_observed=obs, optimism_se=se,
                         icc_mean=icc_sum.loc[k, "icc_mean"], icc_median=icc_sum.loc[k, "icc_median"],
                         icc_iqr=f"{icc_sum.loc[k, 'icc_q25']:.3f}-{icc_sum.loc[k, 'icc_q75']:.3f}",
                         true_auc_used=ta, optimism_pred_icc_mean=preds["icc_mean"],
                         optimism_pred_icc_median=preds["icc_median"],
                         optimism_pred_icc_obs_mean=preds["icc_obs_mean"],
                         pred_minus_obs=preds["icc_mean"] - obs, note="; ".join(note)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"table_{RUN_ID}.csv", index=False)
    cols = ["k", "n", "model", "auc_row_mean", "auc_patient_mean", "optimism_observed", "optimism_se",
            "icc_mean", "icc_median", "icc_iqr", "true_auc_used", "optimism_pred_icc_mean", "pred_minus_obs", "note"]
    def to_md(t):
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
        head = "| " + " | ".join(t.columns) + " |\n|" + "|".join("---" for _ in t.columns) + "|"
        return head + "\n" + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in t.itertuples(index=False))
    md = to_md(tab[cols])
    (out / f"table_{RUN_ID}.md").write_text(f"run_id {RUN_ID}, git {ghash}\n\n{md}\n")
    print(md, flush=True)

    # figure
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    mk = {"lr": "o", "knn": "s", "rf": "^"}; col = {4: "#0072B2", 8: "#D55E00"}
    for _, r in tab.iterrows():
        if np.isnan(r.optimism_pred_icc_mean):
            continue
        ax.errorbar(r.optimism_pred_icc_mean, r.optimism_observed, yerr=1.96 * r.optimism_se,
                    fmt=mk[r.model], color=col[r.k], ms=6, capsize=2, lw=1)
        ax.annotate(str(r.n), (r.optimism_pred_icc_mean, r.optimism_observed), fontsize=6,
                    xytext=(4, 2), textcoords="offset points")
    lim = (min(0, tab.optimism_pred_icc_mean.min(), tab.optimism_observed.min()) - 0.02,
           max(tab.optimism_pred_icc_mean.max(), tab.optimism_observed.max()) + 0.05)
    ax.plot(lim, lim, color="black", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted optimism (v1 calculator)"); ax.set_ylabel("observed optimism (row - patient CV)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk[m], color="grey", ls="", label=m.upper()) for m in MODELS]
    h += [Line2D([], [], color=col[k], lw=2, label=f"k = {k}") for k in cohorts]
    ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Observed vs predicted optimism, CinC 2019 set A", fontsize=9)
    fig.text(0.01, 0.005, f"run_id {RUN_ID} | git {ghash[:10]} | {a.n_sub} subsamples/point; bars 95% CI",
             fontsize=6, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"fig_obs_vs_pred_{RUN_ID}.{ext}", dpi=600 if ext == "png" else None)

    # limitations with the numbers behind them
    sel = ftab[ftab.selected]
    lab_on = {k: float((cohorts[k][cohorts[k].y_pat == 1].groupby("pid").SepsisLabel.max() == 1).mean()) for k in cohorts}
    corr = cohorts[4][feats].corr().abs().to_numpy(); mean_abs_r = float(corr[np.triu_indices(P, 1)].mean())
    lim_lines = [
        f"Non-Gaussian features: {int((sel.skewness_observed.abs() > 1).sum())}/{P} selected features have |skewness| > 1 "
        f"(max {sel.skewness_observed.abs().max():.1f}, {sel.skewness_observed.abs().idxmax()}); the simulation uses Gaussian features.",
        f"Time-varying label: the patient label aggregates a label that switches on during the stay; among septic patients the "
        f"label is already active within the first 4 rows for {lab_on.get(4, float('nan')):.1%}"
        + (f" (first 8 rows: {lab_on[8]:.1%})" if 8 in lab_on else "") + "; for the rest the modelled rows precede onset.",
        f"Missingness: selected features are missing in {sel.missingness.min():.1%}-{sel.missingness.max():.1%} of rows "
        f"(median {sel.missingness.median():.1%}); training-fold median imputation repeats one value across a patient's rows, "
        f"which changes ICC, here lowering it (imputed vs observed-only ICC in icc_per_feature.csv); the simulation has no missingness.",
        f"Correlated features: mean |r| between selected features is {mean_abs_r:.2f}; the simulation uses independent features.",
        f"Heterogeneous ICC: per-feature ICC ranges " + "; ".join(
            f"k={k}: {icc.loc[icc.k == k, 'icc_imputed'].min():.2f}-{icc.loc[icc.k == k, 'icc_imputed'].max():.2f}" for k in cohorts)
        + "; the simulation uses one ICC for all features and the calculator takes the mean.",
        "Consecutive hourly rows are autocorrelated and trending, not exchangeable draws around a patient mean as in the simulation.",
        "Class balance was forced to 50/50 by subsampling; the full cohort is imbalanced.",
        "The calculator's true_auc input is the observed patient-split AUC (a learned, finite-sample estimate), whereas in the "
        "simulation true_auc is the oracle AUC; the input was clipped to the grid where noted.",
    ]
    (out / "limitations.md").write_text(f"# Limitations observed in the real-data demonstration ({RUN_ID}, git {ghash})\n\n"
                                        + "\n".join(f"- {l}" for l in lim_lines) + "\n")
    summary = dict(run_id=RUN_ID, git_hash=ghash, data=a.data, n_subsamples=a.n_sub, features=feats,
                   cohorts={k: infos[k] for k in cohorts}, icc_summary=icc_sum.reset_index().to_dict("records"),
                   skipped_k=[k for k in KS if k not in cohorts])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for k in KS:
        if k in infos:
            print(f"k = {k}: {infos[k]['frac_excluded']:.1%} of {infos[k]['n_patients_total']} patients excluded (< {k} rows); "
                  f"eligible sepsis {infos[k]['n_eligible_sepsis']}, non-sepsis {infos[k]['n_eligible_nonsepsis']}")
    print("\n".join(lim_lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

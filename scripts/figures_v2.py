"""Manuscript figures from an LR+kNN (or full) results.csv, without touching src/.

  Figure 1 at p = 20 and p = 180 (same layout as mie27.figures.fig1)
  Figure 2 with fixed axes and the patient-split rate in grey
  PDF + SVG (vector) and PNG at 600 dpi (for Word); CSV of Figure 2 values;
  diagnostics: p = 20 graded-vs-saturated table, control-ICC homogeneity,
  S1 kNN optimism by ICC.

    python scripts/figures_v2.py --results early/v1/results.csv --outdir early/v1/figs_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from mie27 import figures as F  # noqa: E402

FMTS = ("pdf", "svg", "png")
N_TICKS = [20, 30, 50, 100, 200]
PANELS = [s for s, _ in F.FIG1_PANELS]
# ICC 0 moved off grey (grey now means "patient split"); Okabe-Ito palette
ICC_COLORS = {0.0: "#CC79A7", 0.3: "#56B4E9", 0.6: "#0072B2", 0.9: "#D55E00"}


def save(fig, base: Path):
    for f in FMTS:
        fig.savefig(base.with_suffix(f".{f}"), dpi=600 if f == "png" else None, bbox_inches="tight")


def caption(res, extra):
    return (f"run_id {res.run_id.iloc[0]} | git {str(res.git_hash.iloc[0])[:10]} | "
            f"{int(res.n_reps.min())} replicates/cell | {extra}")


def fig1(res, outdir, models, p, auc, tag, extra):
    import matplotlib.figure as mfig
    orig, orig_sup = F._stamp, mfig.Figure.suptitle
    F._stamp = lambda fig, draft, run_id, ghash, n_reps: fig.text(
        0.01, 0.003, caption(res, extra), fontsize=7, color="#444444")
    # neutral title: the src title states a conclusion ("grows with ICC") the data may not support
    mfig.Figure.suptitle = lambda self, t, **kw: orig_sup(
        self, f"Optimism of row-level over patient-level CV by patient ICC "
              f"(p = {p}, true AUC = {auc:.2f}; bars = 95% Monte Carlo interval)", **kw)
    try:
        base = outdir / f"fig1_nomogram_p{p}_{tag}"
        for f in FMTS:
            plt.rcParams["savefig.dpi"] = 600 if f == "png" else "figure"
            F.fig1(res, base.with_suffix(f".{f}"), models, p, auc, draft=False)
    finally:
        F._stamp, mfig.Figure.suptitle = orig, orig_sup
    return base


def fig2(res, outdir, models, p, tag, extra):
    d = res[(res.p == p) & (res.true_auc == 0.5)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True, sharey=True)
    for ax, sc in zip(axes.flat, F.FIG2_ORDER):
        g = d[d.scenario == sc]
        for m in models:
            c = g[(g.split == "c_patient") & (g.model == m)].groupby("n").false_alarm_070
            lo, hi, mean = c.min(), c.max(), c.mean()
            ax.fill_between(mean.index, lo.values, hi.values, color="#BBBBBB", alpha=0.35, lw=0)
            ax.plot(mean.index, mean.values, F.MODEL_STYLE[m], color="#888888", lw=1.4)
            for icc, gg in g[(g.split == "a_row") & (g.model == m)].groupby("icc"):
                gg = gg.sort_values("n")
                ax.plot(gg.n, gg.false_alarm_070, F.MODEL_STYLE[m], color=ICC_COLORS[icc],
                        marker="o", ms=3, lw=1.3)
        ax.set_title(sc + ("  (ICC has no effect: k = 1)" if sc == "control" else ""), fontsize=8.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(N_TICKS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(18, 220)
        ax.set_ylim(-0.02, 1.02)
    fig.supxlabel("patients (n)", fontsize=9, y=0.085)
    fig.supylabel("P(cross-validated AUC > 0.70 | true AUC 0.50)", fontsize=9)
    h = [Line2D([], [], color=c, lw=2, label=f"row split, ICC {k:g}") for k, c in ICC_COLORS.items()]
    h += [Line2D([], [], color="#888888", lw=1.4, label="patient split (mean over ICC)"),
          Patch(color="#BBBBBB", alpha=0.35, label="patient split, range over ICC")]
    h += [Line2D([], [], color="black", ls=F.MODEL_STYLE[m], label=F.MODEL_LABEL[m]) for m in models]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Null false-alarm rate, row-level vs patient-level CV (p = {p})", fontsize=10)
    fig.text(0.01, -0.035, caption(res, extra), fontsize=7, color="#444444")
    fig.tight_layout(rect=(0.02, 0.11, 1, 0.96))
    base = outdir / f"fig2_null_p{p}_{tag}"
    save(fig, base)
    plt.close(fig)
    return base


def fig2_csv(res, outdir, tag):
    d = res[(res.true_auc == 0.5) & res.split.isin(["a_row", "c_patient"])].copy()
    d["false_alarm_070_mcse"] = np.sqrt(d.false_alarm_070 * (1 - d.false_alarm_070) / d.n_reps)
    cols = ["run_id", "git_hash", "model", "scenario", "p", "icc", "n", "split", "n_reps",
            "false_alarm_070", "false_alarm_070_mcse", "false_alarm_070_pooled"]
    path = outdir / f"fig2_values_{tag}.csv"
    d[cols].sort_values(["model", "p", "scenario", "icc", "n", "split"]).to_csv(path, index=False)
    return path


def graded_table(res, outdir, models, auc, tag):
    """Pre-stated rule, per (model, panel, n) at p = 20: 'graded' if optimism rises by
    > 2 MC SE at each step ICC 0->0.3->0.6->0.9; 'saturated' if row-split AUC >= 0.98 at
    ICC 0.6 or the 0.6->0.9 step is within 2 SE; otherwise 'partly graded'."""
    d = res[(res.split == "a_row") & (res.p == 20) & (res.true_auc == auc) & res.scenario.isin(PANELS)]
    rows = []
    for (m, sc, n), g in d.groupby(["model", "scenario", "n"]):
        if m not in models:
            continue
        g = g.set_index("icc").sort_index()
        o, se, a = g.optimism_mean, g.optimism_mcse, g.auc_mean
        steps = [(o.iloc[i + 1] - o.iloc[i]) / np.hypot(se.iloc[i + 1], se.iloc[i]) for i in range(3)]
        if a.loc[0.6] >= 0.98 or abs(steps[2]) < 2:
            v = "saturated"
        elif all(s > 2 for s in steps):
            v = "graded"
        else:
            v = "partly graded"
        rows.append(dict(run_id=g.run_id.iloc[0], model=m, scenario=sc, n=n, p=20, true_auc=auc,
                         **{f"opt_icc{k:g}": o.loc[k] for k in o.index},
                         **{f"auc_row_icc{k:g}": a.loc[k] for k in a.index},
                         z_step_0_03=steps[0], z_step_03_06=steps[1], z_step_06_09=steps[2], verdict=v))
    t = pd.DataFrame(rows)
    t.to_csv(outdir / f"p20_graded_vs_saturated_{tag}.csv", index=False)
    return t


def control_homogeneity(res):
    from scipy.stats import chi2_contingency
    d = res[(res.scenario == "control") & (res.true_auc == 0.5) & (res.split == "a_row")]
    out = []
    for (m, p, n), g in d.groupby(["model", "p", "n"]):
        k = np.round(g.false_alarm_070 * g.n_reps).astype(int).to_numpy()
        tab = np.vstack([k, g.n_reps.to_numpy() - k])
        pval = 1.0 if tab[0].sum() in (0, tab.sum()) else chi2_contingency(tab)[1]
        out.append(dict(model=m, p=p, n=n, rates=" / ".join(f"{x:.3f}" for x in g.sort_values("icc").false_alarm_070),
                        p_value=pval))
    t = pd.DataFrame(out)
    t["p_holm"] = np.minimum(1, np.maximum.accumulate(np.sort(t.p_value) * (len(t) - np.arange(len(t)))))[
        np.argsort(np.argsort(t.p_value))]
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="early/v1/results.csv")
    ap.add_argument("--outdir", default="early/v1/figs_v2")
    ap.add_argument("--models", nargs="+", default=["lr", "knn"])
    ap.add_argument("--true-auc", type=float, default=0.65)
    ap.add_argument("--note", default="LR + kNN; RF pending")
    a = ap.parse_args(argv)
    res = pd.read_csv(a.results)
    assert res.n_reps.min() >= 200, "fewer than 200 replicates in some cell"
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.models) + "_" + res.run_id.iloc[0]
    F._style()
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    for p in (20, 180):
        print("wrote", fig1(res, out, a.models, p, a.true_auc, tag, a.note), "{.pdf,.svg,.png}")
    for p in (180, 20):
        print("wrote", fig2(res, out, a.models, p, tag, a.note), "{.pdf,.svg,.png}")
    print("wrote", fig2_csv(res, out, tag))

    t = graded_table(res, out, a.models, a.true_auc, tag)
    print(f"\n(1) p = 20, true AUC {a.true_auc}: graded vs saturated (rule in docstring)")
    print(t.groupby(["model", "scenario"]).verdict.value_counts().unstack(fill_value=0).to_string())
    print(t[["model", "scenario", "n", "opt_icc0", "opt_icc0.3", "opt_icc0.6", "opt_icc0.9",
             "auc_row_icc0.9", "z_step_06_09", "verdict"]].to_string(index=False, float_format="%.3f"))

    s1 = res[(res.split == "a_row") & (res.model == "knn") & res.scenario.str.startswith("S1")
             & (res.true_auc == a.true_auc)]
    piv = s1.pivot_table(index=["scenario", "p", "n"], columns="icc", values="optimism_mean")
    print(f"\n(2) kNN S1 optimism by ICC (true AUC {a.true_auc}); change 0.6 -> 0.9 in last column")
    piv["d_06_09"] = piv[0.9] - piv[0.6]
    print(piv.to_string(float_format="%.3f"))

    h = control_homogeneity(res)
    print("\n(4) control: false-alarm rate across ICC 0/0.3/0.6/0.9, chi-square homogeneity, Holm over "
          f"{len(h)} tests")
    print(h.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()

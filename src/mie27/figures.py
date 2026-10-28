"""fig1_nomogram.pdf and fig2_null.pdf.

Refuses to build unless runs/<run_id>/acceptance.json records that every
acceptance test passed. --draft overrides and stamps every page DRAFT.

    python -m mie27.figures --run-id v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

N_COLORS = {20: "#0072B2", 30: "#56B4E9", 50: "#009E73", 100: "#E69F00", 200: "#D55E00"}
ICC_COLORS = {0.0: "#999999", 0.3: "#56B4E9", 0.6: "#0072B2", 0.9: "#D55E00"}
MODEL_LABEL = {"lr": "L2 logistic regression", "knn": "kNN (k = 5)", "rf": "Random forest"}
MODEL_STYLE = {"lr": "-", "knn": "--", "rf": ":"}
FIG1_PANELS = [("S1_r2_d0", "S1 eyes: 2 eyes x 2 repeats (k = 4, d = 0)"),
               ("S2_k4", "S2 imaging: k = 4 visits"),
               ("S3", "S3 EHR: k ~ 1 + NB(4, 1), cap 50")]
FIG2_ORDER = ["control", "S1_r1_d0", "S1_r1_d0.2", "S1_r2_d0", "S1_r2_d0.2", "S2_k4", "S2_k8", "S3"]


def _style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})


def _stamp(fig, draft: bool, run_id: str, ghash: str, n_reps: int):
    fig.text(0.01, 0.005, f"run {run_id} | git {ghash[:10]} | {n_reps} replicates/cell",
             fontsize=6, color="#666666")
    if draft:
        fig.text(0.5, 0.5, "DRAFT - acceptance tests not passed", fontsize=28, color="red",
                 alpha=0.25, ha="center", va="center", rotation=25, transform=fig.transFigure)


def fig1(res: pd.DataFrame, out: Path, models, p: int, true_auc: float, draft: bool):
    d = res[(res.split == "a_row") & (res.p == p) & (res.true_auc == true_auc)]
    fig, axes = plt.subplots(len(models), 3, figsize=(9, 2.8 * len(models)),
                             sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(models):
        for j, (sc, title) in enumerate(FIG1_PANELS):
            ax = axes[i, j]
            g = d[(d.model == m) & (d.scenario == sc)]
            for n, gn in g.groupby("n"):
                gn = gn.sort_values("icc")
                ax.errorbar(gn.icc, gn.optimism_mean, yerr=1.96 * gn.optimism_mcse,
                            color=N_COLORS[n], marker="o", ms=3, lw=1.4, capsize=2,
                            label=f"n = {n}")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(title if i == 0 else "")
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABEL[m]}\noptimism, AUC(a) - AUC(c)")
            if i == len(models) - 1:
                ax.set_xlabel("patient ICC")
            ax.set_xticks([0, 0.3, 0.6, 0.9])
    axes[0, -1].legend(title="patients", frameon=False, loc="upper left")
    fig.suptitle(f"Row-level CV optimism grows with ICC and shrinks with n "
                 f"(p = {p}, true AUC = {true_auc:.2f}; bars = 95% Monte Carlo interval)",
                 fontsize=10)
    _stamp(fig, draft, *_meta(res))
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)


def fig2(res: pd.DataFrame, out: Path, models, p: int, draft: bool):
    d = res[(res.split == "a_row") & (res.p == p) & (res.true_auc == 0.5)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, sc in zip(axes.flat, FIG2_ORDER):
        g = d[d.scenario == sc]
        for (m, icc), gg in g.groupby(["model", "icc"]):
            if m not in models:
                continue
            gg = gg.sort_values("n")
            ax.plot(gg.n, gg.false_alarm_070, MODEL_STYLE[m], color=ICC_COLORS[icc],
                    marker="o", ms=3, lw=1.3)
        ax.set_title(sc)
        ax.set_xscale("log")
        ax.set_xticks([20, 30, 50, 100, 200])
        ax.set_xticklabels(["20", "30", "50", "100", "200"])
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(AUC(a) > 0.70 | true AUC 0.50)")
    for ax in axes[1, :]:
        ax.set_xlabel("patients (n)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=2, label=f"ICC {k:g}") for k, c in ICC_COLORS.items()]
    h += [Line2D([], [], color="black", ls=MODEL_STYLE[m], label=MODEL_LABEL[m]) for m in models]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False)
    fig.suptitle(f"Null false-alarm rate under row-level CV (p = {p})", fontsize=10)
    _stamp(fig, draft, *_meta(res))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def _meta(res):
    return res.run_id.iloc[0], str(res.git_hash.iloc[0]), int(res.n_reps.min())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models", nargs="+", default=["lr", "knn"])
    ap.add_argument("--p", type=int, default=180)
    ap.add_argument("--true-auc", type=float, default=0.65, help="fig1 only")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.runs_dir) / a.run_id
    acc = run_dir / "acceptance.json"
    passed = acc.exists() and json.loads(acc.read_text()).get("all_passed", False)
    if not passed and not a.draft:
        sys.exit(f"Refusing: acceptance tests have not all passed for {a.run_id} "
                 f"(see {acc}). Use --draft for a watermarked preview.")
    _style()
    res = pd.read_csv(run_dir / "results.csv")
    sfx = "" if passed else "_DRAFT"
    fig1(res, run_dir / f"fig1_nomogram{sfx}.pdf", a.models, a.p, a.true_auc, not passed)
    fig2(res, run_dir / f"fig2_null{sfx}.pdf", a.models, a.p, not passed)
    print(f"wrote figures to {run_dir} ({'final' if passed else 'DRAFT'})")


if __name__ == "__main__":
    main()

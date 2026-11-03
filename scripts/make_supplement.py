"""Build supplement/supplement.pdf: Figures S1-S3 on one page, captions with run_id and git hash.

    python scripts/make_supplement.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FIGS = [
    ("S1", "figures/figS1_nomogram_p180_v1.png", "v1",
     "Optimism of row-level over patient-level cross-validation (difference in fold-mean AUC) against "
     "patient ICC, one line per number of patients. Columns: S1 (two eyes x two repeats, d = 0), S2 (four "
     "visits), S3 (EHR, rows per patient 1 + NB(4, 1)); rows: LR, kNN, RF; p = 180, true AUC 0.65; bars 95% "
     "Monte Carlo intervals; 200 replicates per point."),
    ("S2", "figures/figS2_null_p180_v1.png", "v1",
     "Probability that cross-validated AUC exceeded 0.70 when the true AUC was 0.50, against number of "
     "patients, by scenario variant. Coloured lines: row-level split, one colour per ICC; grey line and band: "
     "patient-level split, mean and range over ICC; line style: classifier; p = 180; 200 replicates per point."),
    ("S3", "figures/figS3_obs_vs_pred_realdemo-v1.png", "realdemo-v1",
     "Observed optimism (row-level minus patient-level cross-validated AUC) against the calculator's "
     "prediction on the PhysioNet/CinC 2019 sepsis challenge training set A; k = 4 and 8 hourly rows per "
     "patient; labels give n; bars 95% CI over 100 class-balanced subsamples; line is identity."),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="supplement/supplement.pdf")
    ap.add_argument("--hashes", default="supplement/figure_provenance.json")
    a = ap.parse_args(argv)
    hashes = json.loads(Path(a.hashes).read_text()) if Path(a.hashes).exists() else {}
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.05, 0.985, "Supplement: Many Rows, Few Patients. Figures S1-S3.", fontsize=8, va="top")
    tops = [0.955, 0.625, 0.325]; heights = [0.29, 0.25, 0.23]
    for (lab, path, run, cap), top, h in zip(FIGS, tops, heights):
        img = mpimg.imread(path)
        w = min(0.92, h * (img.shape[1] / img.shape[0]) * (11.69 / 8.27))
        ax = fig.add_axes([(1 - w) / 2, top - h, w, h]); ax.imshow(img); ax.axis("off")
        prov = f"run_id {run}" + (f", git {hashes[run][:10]}" if run in hashes else "")
        fig.text(0.05, top - h - 0.01, f"Figure {lab}. {cap} ({prov}.)", fontsize=6.5, wrap=True, va="top")
    fig.text(0.05, 0.012, "Repository: https://github.com/riziamm/data-leakage-study", fontsize=6.5, color="#444444")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

# Sample-Unit Leakage in Clinical ML

Code, experiment pipelines, and empirical optimism calculator accompanying:
"Many Rows, Few Patients: A Simulation and Nomogram for Sample-Unit 
Leakage in Clinical Machine Learning" (MIE 2027).

Target Paper:  MIE 2027 (Tag: mie2027-submission)
Simulation:    Tag: v1-sim | Commit: b80d4a3
Environment:   Python 3.11 | uv | requirements.lock
License:       MIT
Documentation: DATA.md (external datasets) | DECISIONS.md (protocol audit log)

Simulation experiments use fully reproducible synthetic data. 
The clinical validation benchmark relies on external data 
described in DATA.md. For supplementary figures and extended proofs, 
see supplement/supplement.pdf.

Compute: run v1 used 1268 core-hours for RF, 21.6 for LR and 7.5 for kNN on
ANU School of Computing, Neuroinformatics Cluster, one 250-core CPU node. 
The real-data check took 3 minutes on 32 cores.

## Quickstart: Optimism Calculator
The calculator estimates expected AUC inflation (row-level CV minus patient-level CV)
without rerunning the Monte Carlo grid:

```
import sys
sys.path.insert(0, "runs/v1")
from calculator import optimism

# Estimate expected AUC optimism
delta_auc = optimism(
    n=100,         # Cohort size (patients)
    k_mean=4,      # Mean records per patient
    icc=0.6,       # Intraclass correlation coefficient (within-patient clustering)
    p=180,         # Feature dimensions
    model="knn",   # Classifier: 'lr' | 'knn' | 'rf'
    true_auc=0.65  # True underlying patient-level AUC
)

print(f"Predicted Optimism: +{delta_auc:.3f} AUC")

```
### Constraints: Strictly bound to the experimental parameter grid; raises ValueError
on out-of-distribution inputs.
### Accuracy: Held-out RMSE across the parameter grid is $0.013\text{--}0.023$ 
AUC (worst single-cell error: $0.080$; see runs/v1/calculator_report.txt).

## Layout

    src/mie27/         generate, calibrate, cv, run, summarise, build_calculator,
                       figures, fast (kNN and grouped-fold replicas, tested
                       identical to scikit-learn)
    scripts/           test3_comparator.py, figures_v2.py, realdemo.py,
                       make_supplement.py
    jobs/              launch_local.sh
    tests/             acceptance tests 0-5
    runs/v1/           results.csv, calibration.csv, test_output.txt,
                       test3_comparator80.csv, calculator.py,
                       calculator_report.txt, final_numbers.txt,
                       figs_v2/fig2_values_lr_knn_rf_v1.csv
    runs/realdemo-v1/  table, per-feature ICC and missingness, limitations.md
    figures/           Figures 1-2 and S1-S3, PNG and PDF
    supplement/        supplement.pdf 
    DECISIONS.md       deviations from the protocol and why
    DATA.md            external dataset, licence, what is stored

## Reproduce

Wall times are for one 250-core node.

    # 1. environment setup, ~2 min
    uv venv -p 3.11 .venv && . .venv/bin/activate
    uv pip install -r requirements.lock && uv pip install -e .

    # 2. LR and kNN, 960 cells x 200 replicates, ~5 min
    jobs/launch_local.sh v1 250 "lr knn"

    # 3. RF, ~5 h; then merge into v1
    CHUNKS=10 jobs/launch_local.sh v1-rf 250 "rf"
    cp -r runs/v1-rf/cells/rf runs/v1/cells/

    # 4. results, test-3 comparator, acceptance tests, ~25 min
    python -m mie27.summarise --run-id v1
    python scripts/test3_comparator.py --run-dir runs/v1 --jobs 16
    MIE27_RUN_DIR=runs/v1 python -m pytest -v tests | tee runs/v1/test_output.txt

    # 5. calculator and figures, ~2 min
    python -m mie27.build_calculator --run-id v1 | tee runs/v1/calculator_report.txt
    python scripts/figures_v2.py --results runs/v1/results.csv \
        --outdir runs/v1/figs_v2 --models lr knn rf --note "LR, kNN, RF"

    # 6. real-data check (download per DATA.md, 10 min) and supplement, ~3 min
    python scripts/realdemo.py --data data/training_setA --jobs 32
    python scripts/make_supplement.py

Reproducibility is ensured given every cell's seed is a hash of (base seed 20271, cell id)
and all classifiers see the same data and folds. Runs are resumable.

## Figures

| paper | file | run_id | code |
|---|---|---|---|
| Figure 1 | figures/fig1_nomogram_p20_v1 | v1 | b80d4a3 |
| Figure 2 | figures/fig2_null_p20_v1 | v1 | b80d4a3 |
| Figure S1 (p = 180 nomogram) | figures/figS1_nomogram_p180_v1 | v1 | b80d4a3 |
| Figure S2 (p = 180 null) | figures/figS2_null_p180_v1 | v1 | b80d4a3 |
| Figure S3 (CinC 2019) | figures/figS3_obs_vs_pred_realdemo-v1 | realdemo-v1 | see supplement/figure_provenance.json |

## Calculator

    import sys; sys.path.insert(0, "runs/v1")
    from calculator import optimism
    optimism(n=100, k_mean=4, icc=0.6, p=180, model="knn", true_auc=0.65)

Returns the expected row-level minus patient-level CV AUC. Refuses inputs
outside the simulated grid. Held-out RMSE 0.013-0.023, worst cell 0.08
(runs/v1/calculator_report.txt).

## results.csv

One row per cell x classifier x split (a_row, b_unit, c_patient). auc_mean and
quantiles are fold-mean CV AUC over 200 replicates. optimism_mean is this split
minus c_patient. false_alarm_070 is P(AUC > 0.70) in true-AUC-0.50 cells.
*_pooled columns use pooled out-of-fold AUC (sensitivity analysis).
auc_indep_mean is the AUC of a model trained on all rows and scored on 2000
independent patients.

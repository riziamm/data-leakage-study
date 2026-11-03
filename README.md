# data-leakage-study

Code, results and calculator for "Many Rows, Few Patients: A Simulation and
Nomogram for Sample-Unit Leakage in Clinical Machine Learning" (MIE 2027).
Simulation data are synthetic. The real-data check uses a public dataset that
is not redistributed here (DATA.md). MIT licence.

Simulation code: tag v1-sim, commit b80d4a3. Submitted state: tag
mie2027-submission. Python 3.11 (3.11.15 at pinning, 3.11.18 on the run node),
scikit-learn 1.9.1, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1, matplotlib
3.11.2, joblib 1.6.0; pins in requirements.lock.

Compute: run v1 used 1268 core-hours for RF, 21.6 for LR and 7.5 for kNN on
ANU School of Computing, Neuroinformatics, one 250-core CPU node. The real-data check took 3 minutes on
32 cores.

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
    figures/           Figures 1-2 and S1-S3, PNG and PDF, named by run_id
    supplement/        supplement.pdf (Figures S1-S3 with captions)
    DECISIONS.md       deviations from the protocol and why
    DATA.md            external dataset, licence, what is stored

## Reproduce

Wall times are for one 250-core node.

    # 1. environment, 2 min
    uv venv -p 3.11 .venv && . .venv/bin/activate
    uv pip install -r requirements.lock && uv pip install -e .

    # 2. LR and kNN, 960 cells x 200 replicates, 7 min
    jobs/launch_local.sh v1 250 "lr knn"

    # 3. RF, 5 h; then merge into v1
    CHUNKS=10 jobs/launch_local.sh v1-rf 250 "rf"
    cp -r runs/v1-rf/cells/rf runs/v1/cells/

    # 4. results, test-3 comparator, acceptance tests, 25 min
    python -m mie27.summarise --run-id v1
    python scripts/test3_comparator.py --run-dir runs/v1 --jobs 16
    MIE27_RUN_DIR=runs/v1 python -m pytest -v tests | tee runs/v1/test_output.txt

    # 5. calculator and figures, 2 min
    python -m mie27.build_calculator --run-id v1 | tee runs/v1/calculator_report.txt
    python scripts/figures_v2.py --results runs/v1/results.csv \
        --outdir runs/v1/figs_v2 --models lr knn rf --note "LR, kNN, RF"

    # 6. real-data check (download per DATA.md, 10 min) and supplement, 3 min
    python scripts/realdemo.py --data data/training_setA --jobs 32
    python scripts/make_supplement.py

Every cell's seed is a hash of (base seed 20271, cell id), so any machine or
shard gives the same numbers and all classifiers see the same data and folds.
Interrupted runs resume.

## Figures

| paper | file | run_id | code |
|---|---|---|---|
| Figure 1 | figures/fig1_nomogram_p20_v1 | v1 | b80d4a3 |
| Figure 2 | figures/fig2_null_p20_v1 | v1 | b80d4a3 |
| Figure S1 (p = 180 nomogram) | figures/figS1_nomogram_p180_v1 | v1 | b80d4a3 |
| Figure S2 (p = 180 null) | figures/figS2_null_p180_v1 | v1 | b80d4a3 |
| Figure S3 (CinC 2019) | figures/figS3_obs_vs_pred_realdemo-v1 | realdemo-v1 | see supplement/figure_provenance.json |

The supplement the paper cites is supplement/supplement.pdf.

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

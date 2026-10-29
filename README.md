# data-leakage-study

Simulation of how much row-level cross-validation overstates AUC when patients
contribute several correlated rows (eyes, imaging visits, EHR encounters), plus
a calculator fitted to the results and one real-data check of that calculator.
All simulation data are synthetic. Python 3.11.

Read DECISIONS.md before using any number: it records where the original spec
was ambiguous or impossible and what was done instead.

## Layout

    src/mie27/          simulation: generate, calibrate, cv, run, summarise,
                        figures, build_calculator, fast (tested kNN/fold replicas)
    scripts/            analysis scripts that do not touch src/:
                        test3_comparator.py, figures_v2.py, realdemo.py
    jobs/               launch_local.sh (grid runner); ops scripts used for v1,
                        kept as a record
    tests/              acceptance tests 0-5
    docs/               methods text, drafter brief, patches
    runs/v1/            results.csv, test output, calculator.py, figs_v2/, logs/
    runs/realdemo-v1/   real-data check (see DATA.md)
    DATA.md             external dataset: source, licence, what is stored

## Runs

Simulation code for every v1 number is commit b80d4a3 (tag v1-sim).
960 cells x 3 classifiers x 200 replicates. Per-cell seeds are hashed from
(base seed 20271, cell id), so any machine or shard reproduces the same numbers
and all classifiers see identical data and folds.

- v1: LR, kNN, RF. results.csv, figures, calculator.
- v1-rf: where RF ran before being merged into v1.
- v1-ctrl-ext: 200 extra replicates of one control cell (see DECISIONS D7).
- realdemo-v1: calculator check on PhysioNet/CinC 2019 set A.

## Reproduce

    uv venv -p 3.11 .venv && . .venv/bin/activate
    uv pip install -r requirements.lock && uv pip install -e .

    jobs/launch_local.sh v1 250 "lr knn"              # ~15 min on 250 cores
    CHUNKS=10 jobs/launch_local.sh v1-rf 250 "rf"     # ~5 h on 250 cores
    cp -r runs/v1-rf/cells/rf runs/v1/cells/
    python -m mie27.summarise --run-id v1
    python scripts/test3_comparator.py --run-dir runs/v1 --jobs 16
    MIE27_RUN_DIR=runs/v1 python -m pytest -v tests | tee runs/v1/test_output.txt
    python -m mie27.build_calculator --run-id v1 | tee runs/v1/calculator_report.txt
    python scripts/figures_v2.py --results runs/v1/results.csv \
        --outdir runs/v1/figs_v2 --models lr knn rf --note "LR, kNN, RF"

    # real-data check (download instructions in DATA.md)
    python scripts/realdemo.py --data data/training_setA --jobs 32

Figures for the paper are the ones in runs/v1/figs_v2/. The fig1_nomogram.pdf
and fig2_null.pdf written by mie27.figures are earlier versions; do not use them.

Runs resume: rerun the same launch command and finished replicates are kept.
Workers need about 0.25 GB each.

## Calculator

    import sys; sys.path.insert(0, "runs/v1")
    from calculator import optimism
    optimism(n=100, k_mean=4, icc=0.6, p=180, model="knn", true_auc=0.65)

Returns expected row-level minus patient-level CV AUC. Refuses inputs outside
the simulated grid. Held-out error is in runs/v1/calculator_report.txt
(RMSE 0.013-0.023, worst cell 0.08).

## results.csv

One row per cell x classifier x split (a_row, b_unit, c_patient). auc_mean and
quantiles are fold-mean CV AUC over 200 replicates. optimism_mean is this split
minus c_patient; the a_row row is the headline optimism. false_alarm_070 is
P(AUC > 0.70) in true-AUC-0.50 cells. *_pooled columns use pooled out-of-fold
AUC (sensitivity analysis). auc_indep_mean is the AUC of a model trained on all
rows and scored on 2000 independent patients.

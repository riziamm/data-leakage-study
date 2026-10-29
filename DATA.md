# External data

The simulation uses no real data. One public dataset is used for the real-data
check of the calculator (run realdemo-v1).

## Source

PhysioNet/CinC Challenge 2019, v1.0.0, training set A. 20,336 ICU patients,
one PSV file per patient, one row per hour, 40 variables plus SepsisLabel.

- https://physionet.org/content/challenge-2019/1.0.0/
- files: https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/
- doi: 10.13026/v64v-d857
- licence text: https://physionet.org/files/challenge-2019/1.0.0/LICENSE.txt

## Licence

CC BY 4.0. Project page: "Anyone can access the files, as long as they conform
to the terms of the specified license." No registration, agreement or fee.
Research use and redistribution are allowed with attribution. Checked
2026-10-29. A copy of LICENSE.txt is kept in runs/realdemo-v1/.

Cite in the paper:

- Reyna M et al. Early Prediction of Sepsis from Clinical Data: The
  PhysioNet/Computing in Cardiology Challenge 2019 (v1.0.0). PhysioNet, 2019.
  doi:10.13026/v64v-d857
- Reyna MA et al. Crit Care Med 2020;48(2):210-217. doi:10.1097/CCM.0000000000004145

## What is in the repo

Raw files are not committed (`data/` is ignored). `runs/realdemo-v1/` holds
derived summaries only: per-feature missingness, skewness, ICC, per-subsample
AUCs, the observed-vs-predicted table and figure.

## Processing

See `scripts/realdemo.py`. Patient label = any SepsisLabel = 1. Features = 20
vitals/labs with least missingness, median-imputed and standardised on the
training fold. Rows = first k hours, k = 4 and 8. Patients with fewer than k
rows are dropped; the fraction is in `summary.json`. Subsamples: n = 50, 100,
200, class-balanced, 100 per n, seed 20271.

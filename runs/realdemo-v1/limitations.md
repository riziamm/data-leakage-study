# Limitations observed in the real-data demonstration (realdemo-v1, git 14a6c0eb1acb8e540e44beda0ec274318e7327be)

- Non-Gaussian features: 11/20 selected features have |skewness| > 1 (max 16.9, WBC); the simulation uses Gaussian features.
- Time-varying label: the patient label aggregates a label that switches on during the stay; among septic patients the label is already active within the first 4 rows for 18.3% (first 8 rows: 26.7%); for the rest the modelled rows precede onset.
- Missingness: selected features are missing in 18.4%-89.8% of rows (median 80.6%); training-fold median imputation repeats one value across a patient's rows, which changes ICC, here lowering it (imputed vs observed-only ICC in icc_per_feature.csv); the simulation has no missingness.
- Correlated features: mean |r| between selected features is 0.13; the simulation uses independent features.
- Heterogeneous ICC: per-feature ICC ranges k=4: 0.02-0.66; k=8: 0.04-0.72; the simulation uses one ICC for all features and the calculator takes the mean.
- Consecutive hourly rows are autocorrelated and trending, not exchangeable draws around a patient mean as in the simulation.
- Class balance was forced to 50/50 by subsampling; the full cohort is imbalanced.
- The calculator's true_auc input is the observed patient-split AUC (a learned, finite-sample estimate), whereas in the simulation true_auc is the oracle AUC; the input was clipped to the grid where noted.

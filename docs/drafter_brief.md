# Drafter brief — unit-leakage simulation, run v1

Every number below comes from run_id **v1** (simulation code git `b80d4a3`, tag
`v1-sim`, repo `riziamm/data-leakage-study`) unless marked otherwise. Quote numbers
with their Monte Carlo SE (MC SE) as given. Do not use anything from `early/`, from
`runs/v1/fig1_nomogram*.pdf` / `fig2_null*.pdf` (superseded), or any earlier run.

## 1. Provenance (for Methods / code availability)
- Fully synthetic data; no patient data used.
- 960 cells (8 scenarios × n {20, 30, 50, 100, 200} × ICC {0, 0.3, 0.6, 0.9} ×
  p {20, 180} × true AUC {0.50, 0.65, 0.80}) × 3 classifiers × 200 replicates.
- Per-cell seeds derived from base seed 20271; all classifiers see identical data
  and folds within a replicate. Python 3.11, scikit-learn 1.9.1.
- Compute: RF 1,268 core-hours, LR 21.6, kNN 7.5, on one 250-core CPU node.
- Methods text: `methods_spec.md` (scenarios, NB(4,1), discordance d, folds,
  true-AUC sweep, classifiers). Calibration: `runs/v1/calibration.csv`.

## 2. Validation (Methods or Supplement)
- All 77 acceptance tests passed (`runs/v1/test_output.txt`).
- No optimism without clustering: at ICC = 0, row-minus-patient optimism pooled
  over 240 cells was −0.00001 (MC SE 0.00013) for kNN, +0.00007 (0.00015) for LR
  and −0.00004 (0.00017) for RF. Cells with |z| ≥ 3: LR 1/240, kNN 0/240, RF 0/240
  (pre-specified limit ≤ 1%).
- Control scenario (one row per patient): cells with |z| ≥ 3: kNN 1/120, RF 1/120,
  LR 0/120. False-alarm rates did not differ across ICC (chi-square homogeneity,
  all 30 Holm-adjusted p ≥ 0.87), as expected since ICC has no effect when k = 1.
- At ICC = 0, n = 200, true AUC 0.80, every split matched each classifier's own
  independent-test AUC (trained on 80% of patients) within 0.006 (limit 0.02).
- One control cell (kNN, n 20, p 20, ICC 0.9, AUC 0.80) had z = −4.05. In 200
  further replicates (run v1-ctrl-ext) it was −0.0048 (MC SE 0.0049, z −0.98);
  combined z −3.54, so the pre-specified rule classifies it as inconclusive. Across
  the other 23 control cells at n = 20, optimism was +0.0019 (0.0010) for kNN and
  +0.0015 (0.0012) for LR: no systematic splitter effect. The RF control cell
  flagged (n 200, p 180, ICC 0.3, z +3.38) was not followed up.

## 3. Results
**Magnitude.** The largest inflation of row-level over patient-level CV occurred
with no true signal (true AUC 0.50) at ICC 0.9:
- kNN +0.510 (MC SE 0.007), S2 k = 8 visits, n 30, p 20
- LR +0.528 (0.014), S3 EHR, n 20, p 180
- RF +0.517 (0.009), S2 k = 8 visits, n 30, p 20

Decomposed as (row − independent-test AUC) + (independent-test − patient-split
AUC), leakage accounted for +0.500, +0.497 and +0.500 respectively; the remainder
(+0.011, +0.031, +0.017) is the patient-level split falling below 0.5 at small n.
Models scored 0.50 on independent data: all apparent skill came from patients
appearing on both sides of the split.

**Dependence on ICC and n** (true AUC 0.65, Figure 1).
- LR: optimism rises with ICC and shrinks with n. Example, S2 k = 4, p = 20:
  ICC 0.3 / 0.6 / 0.9 → +0.158 / +0.288 / +0.426 at n = 20, but
  +0.036 / +0.068 / +0.099 at n = 200.
- kNN and RF: at ICC 0.9 optimism stays around +0.40–0.47 whatever n
  (S2 k = 4, p = 20: kNN +0.402 at n = 20, +0.413 at n = 200; RF +0.441 and +0.401).
  More patients does not protect these classifiers from row-level leakage.
- At p = 20, optimism is graded across ICC for LR and RF in all 15 scenario × n
  combinations; kNN is graded in 10/15 and saturates at small n
  (n 20–30 in S1 and S3, n 20 in S2).
- kNN in S1 (eyes) at p = 180 is non-monotone: optimism is near its maximum by
  ICC 0.3 and falls by up to 0.028 from ICC 0.6 to 0.9. Proposed mechanism (from a
  separate diagnostic simulation, not run v1): at high ICC the fellow-eye and
  repeat rows already fill the neighbour list, and the remaining neighbour slots
  fill with rows of one other patient, whose shared label adds noise.

**False alarms under the null** (Figure 2). Report by scenario and ICC from
`fig2_values_lr_knn_rf_v1.csv`. Pooled over all 320 no-signal cells,
P(row-split AUC > 0.70) was 0.441 (kNN), 0.381 (LR), 0.462 (RF), but this pooled
value depends on the grid's composition and should not be quoted as a single rate.
Even the correct patient-level split is unreliable at n = 20: for LR in S3 at
p = 180 with no signal, P(patient-split AUC > 0.70) = 0.135 and its 95% range was
0.11–0.83.

**AUC estimator.** Pooled out-of-fold AUC fell further below fold-mean AUC in the
patient split than in the row split at ICC = 0 (kNN −0.0050 vs −0.0018; LR −0.0045
vs −0.0020; RF −0.0116 vs −0.0040), so it would have created spurious optimism of
about +0.003 (kNN, LR) and +0.008 (RF). Fold-mean AUC is therefore the primary
estimator; pooled AUC is reported as a sensitivity analysis in results.csv.

**Optimism calculator** (`runs/v1/calculator.py`). Smooth surface in n, mean rows
per patient, ICC and true AUC, fitted per classifier and p on 300 cells each:
control, S1 with one row per eye and no discordance (identical in distribution to
two rows per patient), S2 and S3. Held-out validation (20% of cells): RMSE 0.013–0.023, maximum absolute error
0.041–0.082 (worst: RF, p = 180), against MC SE ≈ 0.003. State it as accurate to
about ±0.02 (RMSE). Valid only inside the simulated grid; it refuses to
extrapolate. Not fitted to the other S1 variants (two repeats per eye, or
discordant eye labels), whose structure is not captured by mean rows per patient.

## 4. Figures
- **Figure 1** (main): `runs/v1/figs_v2/fig1_nomogram_p180_lr_knn_rf_v1.png`.
  Legend: Optimism of row-level over patient-level cross-validation
  (AUC(a) − AUC(c)) against patient ICC, one line per number of patients; columns
  S1 (2 eyes × 2 repeats, d = 0), S2 (4 visits), S3 (k ~ 1 + NB(4,1)); rows LR,
  kNN, RF; p = 180, true AUC 0.65; bars are 95% Monte Carlo intervals; 200
  replicates per point; run v1.
  Supplement: same at p = 20 (`fig1_nomogram_p20_lr_knn_rf_v1.png`).
- **Figure 2** (main): `runs/v1/figs_v2/fig2_null_p180_lr_knn_rf_v1.png`.
  Legend: Probability that cross-validated AUC exceeds 0.70 when the true AUC is
  0.50, against number of patients, by scenario. Coloured lines: row-level split
  by ICC; grey line: patient-level split (mean over ICC), grey band its range over
  ICC; line style marks classifier; p = 180; 200 replicates per point; run v1.
  Values: `fig2_values_lr_knn_rf_v1.csv`. Supplement: p = 20 version.
- For Word use the PNG (600 dpi); keep SVG/PDF for typesetting.

## 5. Deviations from the original protocol (Methods / Limitations)
- S1 variance: ICC_e defined as the eye's share of within-patient variance, because
  ICC_e = ICC_p/2 as a share of total variance exceeds 1 at ICC_p = 0.9.
- "True AUC" = oracle AUC, so β is the same for every p and ICC (β = 0, 0.545, 1.190).
- Test 3 compares each split with an 80%-trained independent-test AUC (the
  100%-trained comparator was biased high by the learning curve).
- Tests 1–2 gate on |z| < 3 per cell with ≤ 1% of cells failing; |optimism| < 0.01
  reported, not gated. A replication test (≥ 200 replicates) was added.
- Fold-mean AUC primary (see AUC estimator above).
- LR max_iter raised to 1000 for convergence at p = 180; otherwise defaults.
- Full rationale: `DECISIONS.md`.

## 6. Files to attach for the drafter
`methods_spec.md`, `DECISIONS.md`, `runs/v1/final_numbers.txt`,
`runs/v1/figs_v2/` (all), `runs/v1/calculator_report.txt`,
`runs/v1/test_output.txt`, `runs/v1/calibration.csv`, `runs/v1/results.csv`,
`README.md` (reproduction commands).

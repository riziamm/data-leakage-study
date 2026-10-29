# Drafter brief

All numbers are from run v1 (simulation code b80d4a3, tag v1-sim, repo
riziamm/data-leakage-study) unless marked realdemo-v1. Give every number its
MC SE as listed. Do not use anything from early/ or from the fig1_nomogram.pdf
and fig2_null.pdf written by mie27.figures.

## Provenance

Synthetic data only. 960 cells (8 scenarios x n {20,30,50,100,200} x ICC
{0,0.3,0.6,0.9} x p {20,180} x true AUC {0.50,0.65,0.80}) x 3 classifiers x
200 replicates. Per-cell seeds from base seed 20271; all classifiers see the
same data and folds. Python 3.11, scikit-learn 1.9.1. Compute: RF 1268
core-hours, LR 21.6, kNN 7.5, one 250-core node.
Methods text: docs/methods_spec.md. Calibration: runs/v1/calibration.csv.

## Validation

All 77 acceptance tests pass (runs/v1/test_output.txt).
At ICC = 0, optimism pooled over 240 cells: kNN -0.00001 (SE 0.00013),
LR +0.00007 (0.00015), RF -0.00004 (0.00017). Cells with |z| >= 3: LR 1/240,
kNN 0, RF 0 (limit 1%). Control: kNN 1/120, RF 1/120, LR 0. Control false-alarm
rates do not differ across ICC (30 chi-square tests, all Holm p >= 0.87).
Test 3: every split within 0.006 of the model's own 80%-trained independent
AUC (limit 0.02). One control cell with z = -4.05 did not replicate
(-0.0048 +/- 0.0049 in 200 more replicates, run v1-ctrl-ext); the other 23
n = 20 control cells pooled show no splitter effect (+0.0019 +/- 0.0010 kNN,
+0.0015 +/- 0.0012 LR). Report it as inconclusive under the pre-set rule.

## Results

Largest inflation, all with no true signal at ICC 0.9:
kNN +0.510 (0.007), S2 k = 8, n 30, p 20;
LR +0.528 (0.014), S3, n 20, p 180;
RF +0.517 (0.009), S2 k = 8, n 30, p 20.
Decomposed as (row - independent AUC) + (independent - patient AUC): leakage
+0.500, +0.497, +0.500; the rest (+0.011, +0.031, +0.017) is the patient split
falling below 0.5 at small n. Models score 0.50 on independent data.

ICC and n (true AUC 0.65, Figure 1). LR optimism rises with ICC and falls with
n: S2 k = 4, p = 20, ICC 0.3/0.6/0.9 gives +0.158/+0.288/+0.426 at n = 20 and
+0.036/+0.068/+0.099 at n = 200. kNN and RF stay near +0.40 to +0.47 at ICC 0.9
for every n (kNN +0.402 at n = 20, +0.413 at n = 200; RF +0.441, +0.401).
More patients do not protect these two. At p = 20 optimism is graded across ICC
for LR and RF in all 15 scenario x n combinations; kNN is graded in 10/15 and
saturates at n = 20-30. At p = 180 kNN in S1 peaks by ICC 0.3 and falls by up
to 0.028 from 0.6 to 0.9 (mechanism in DECISIONS D12, from a separate check).

Null false alarms (Figure 2, values in fig2_values_lr_knn_rf_v1.csv): report by
scenario and ICC. Pooled over all 320 no-signal cells P(row AUC > 0.70) is
0.441 kNN, 0.381 LR, 0.462 RF, but this depends on the grid mix; do not quote
it as a rate. Even the patient split is unreliable at n = 20: LR, S3, p = 180,
no signal, P(patient-split AUC > 0.70) = 0.135, 95% range 0.11-0.83.

Estimator: pooled out-of-fold AUC would have added spurious optimism of about
+0.003 (kNN, LR) and +0.008 (RF) at ICC = 0; fold-mean is primary.

Calculator (runs/v1/calculator.py): held-out RMSE 0.013-0.023, max error
0.04-0.08 (RF, p = 180). State as accurate to about 0.02. Grid only; not fitted
to S1 with repeats or discordant eyes.

Real-data check (realdemo-v1, PhysioNet/CinC 2019 set A, CC BY 4.0, cite per
DATA.md). 20,336 patients, none excluded. Row-level CV inflated AUC by 0.03 to
0.26 depending on classifier, k and n, with patient-split AUC 0.53-0.59.
Calculator predictions were within its stated error for LR (+0.02) and kNN
(-0.05 to +0.01) and under-predicted RF by 0.07-0.10; the trends with k and n
matched. Data violate the simulation assumptions: 11/20 features skewed,
18-90% missingness, label active in the window for 18% (k = 4) and 27% (k = 8)
of septic patients, per-feature ICC 0.02-0.72. One line each in
runs/realdemo-v1/limitations.md.

## Figures

Figure 1: runs/v1/figs_v2/fig1_nomogram_p180_lr_knn_rf_v1.png. Optimism
AUC(a) - AUC(c) vs patient ICC, one line per n; columns S1 (2 eyes x 2 repeats,
d = 0), S2 (4 visits), S3; rows LR, kNN, RF; p = 180, true AUC 0.65; bars 95%
MC intervals; 200 replicates per point. Supplement: p = 20 version.
Figure 2: runs/v1/figs_v2/fig2_null_p180_lr_knn_rf_v1.png. P(CV AUC > 0.70 |
true AUC 0.50) vs n by scenario; colour = ICC (row split); grey = patient split
(mean over ICC, band = range); line style = classifier. Supplement: p = 20.
Figure 3: runs/realdemo-v1/fig_obs_vs_pred_realdemo-v1.png. Observed vs
predicted optimism on real data, 45-degree line, 95% CIs, labels = n.
Use PNG (600 dpi) in Word; SVG/PDF for typesetting.

## Deviations from protocol

DECISIONS.md D1 (S1 variance), D2 (oracle true AUC), D3 (test 3 comparator),
D4 (fold-mean estimator), D7 (z gates, test 0), D9 (LR max_iter), D13
(real-data choices).

## Files

docs/methods_spec.md, DECISIONS.md, DATA.md, runs/v1/final_numbers.txt,
runs/v1/figs_v2/, runs/v1/calculator_report.txt, runs/v1/test_output.txt,
runs/v1/calibration.csv, runs/v1/results.csv, runs/realdemo-v1/.

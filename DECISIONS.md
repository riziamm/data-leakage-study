# Decisions log — mie27-unit-leakage

Each entry: what the spec said, what was done, why. Items marked **NEEDS SIGN-OFF**
change what a reader of the paper would assume; please confirm or overrule.

## D1. S1 nested variance when ICC_p + ICC_e > 1 — ACCEPTED 28 Oct
Spec: patient ICC_p and eye ICC_e, default ICC_e = ICC_p / 2. Taken as shares of
total variance, ICC_p = 0.9 gives 0.9 + 0.45 = 1.35 > 1 (impossible).
Done: ICC_e is the eye's share of the *within-patient* variance:
s2_eye = ICC_e (1 − ICC_p), s2_res = (1 − ICC_p)(1 − ICC_e). Valid on the whole grid.
Consequence: corr(fellow eyes) = ICC_p; corr(repeats of one eye) = ICC_p + s2_eye.

## D2. Meaning of "true AUC" and beta calibration
Done: true AUC = AUC of the oracle score w'x on an independent k = 1 sample.
Since every feature has total variance 1, w'x | y ~ N(beta·y, 1) and
AUC = Φ(beta/√2), so beta = √2 Φ⁻¹(AUC) for **every p and every ICC**.
beta(0.65) = 0.5449, beta(0.80) = 1.1902. "Per p" calibration was still run by
Monte Carlo bisection and stored (calibration.csv); it reproduces the analytic
value, which documents that p does not enter. Test 5 passes for both p.

## D3. Test 3 — ACCEPTED 28 Oct: compare each split with the model's own independent-test AUC
Test 3 asks every split to hit 0.80 ± 0.02 at n = 200, ICC = 0. 0.80 is the
*oracle* AUC (D2). Models trained on 160 rows with 20 or 180 features land well
below it (the learned AUC is below it; see auc_indep_mean in results.csv),
independent of any leakage. Done: each replicate also fits the model on all rows and scores an independent k = 1 sample of 2000 patients (auc_indep); test 3 requires |AUC(split) - auc_indep| <= 0.02 at true AUC 0.80, n = 200, ICC = 0, for every split, model, scenario and p.

## D4. CV AUC estimator: fold-mean (primary), pooled OOF (sensitivity) — ACCEPTED 28 Oct
Pooled out-of-fold AUC is biased *below* 0.5 under the null when fold class
balance varies, and patient-grouped folds vary more (labels cluster in patients).
At ICC = 0, n = 20 this produced spurious "optimism" of +0.013 to +0.019 in
S2_k8 / S3 (≈3 Monte Carlo SE) that is not leakage. Mean of per-fold AUCs removed
it (≈0.000 ± 0.005). Folds with a single class in the test set are skipped and
counted (skipped_folds_mean; ~0.05 per replicate at n = 20, 0 by n = 50).
results.csv keeps both: auc_mean / optimism_mean / false_alarm_070 are fold-mean;
*_pooled columns are pooled OOF.

## D5. Degenerate replicates
Patient labels are redrawn if either class has < 2 patients (otherwise CV AUC is
undefined). Probability ≈ 4e-5 at n = 20; count stored (redraws_total).

## D6. Calculator scope
Fitted on control, S2_k4, S2_k8, S3 (patient → rows). S1 excluded: its eye
nesting and per-eye discordant labels are not a function of k_mean.
Signature optimism(n, k_mean, icc, p, model) plus keyword true_auc (default 0.65,
grid midpoint), because optimism depends on signal strength. p is categorical
{20, 180}: two values cannot support interpolation. Form:
icc·(1 − 1/k)·poly2(icc, ln n, ln k, true_auc) — zero at ICC = 0 and k = 1 by
construction.

## D7. Test 0 added: replication
Tests 1–3 check MC means against ±0.01. At fewer than 200 replicates they are
noise-limited, so test 0 fails any run below 200 replicates/cell and the figure
gate requires all tests to pass. Note even at 200 reps the MC SE of optimism at
n = 20 is ≈ 0.004–0.007, so test 1's ±0.01 spans only ~1.5–2.5 SE across 240+
cells. ACCEPTED 28 Oct: gate is |z| < 3 per cell (z = optimism / MC SE) with at most 1% of cells failing; |optimism| < 0.01 is kept as the reported column optimism_within_001. The same gate is applied to test 2 (same noise argument; absolute column still reported).

## D8. Figure choices
fig1: true AUC 0.65 (option --true-auc), S1 panel = r = 2, d = 0; error bars are
95% Monte Carlo intervals of the mean. fig2: split (a), p = 180, all 8 scenarios,
colour = ICC, line style = model.

## D9. LR max_iter = 1000
sklearn defaults otherwise (L2, C = 1, lbfgs). Default max_iter = 100 does not
always converge at p = 180; raising it changes convergence, not the estimator.

## D10. Random forest — superseded 28 Oct: no Gadi; RF runs as a separate local run (v1-rf)
RF was not run in the build environment (1 CPU core). Measured RF cost here:
10–65 s per replicate at n = 200 (all splits), rough grid total ≈ 900 core-hours
(uncertain by ~2×), ≈ 1,800 SU on Gadi `normal`. The slowest cell
(S2_k8, n = 200, p = 180) is ~3.6 h on one core, which bounds wall time.
Rule: if RF is not complete by 29 Oct 18:00 AEDT, summarise with
`--models lr knn` and record the decision here with the completed-cell count.
Decision: _not yet taken._

## D11. Variance-component assertion (28 Oct)
simulate() asserts patient + eye + residual = 1 on every call; test_3b checks it
over all scenarios x ICC on the grid.

## D12. Execution (28 Oct)
Target is one 250-core node; jobs/launch_local.sh replaces the Gadi PBS scripts.
Measured at the largest cell (S2 k = 8, n = 200, p = 180, ICC 0.9, AUC 0.80),
one worker, peak RSS incl. the independent-test fit: LR 175 MB, kNN 210 MB,
RF 176 MB; time per replicate LR 1.2 s, kNN 0.5 s, RF 130 s. RF cells are split
into replicate chunks (--chunks) so no single task runs for hours; chunked
output is bit-identical to unchunked (verified).

## D13. Calculator surface (28 Oct)
Degree-2 vs degree-3 compared by leave-one-out on a 10-replicate scratch run:
degree 3 is worse (LOO RMSE 0.018→0.018 kNN, 0.028→0.029 LR, max error up), so
degree 2 stays. Held-out (20% of cells) RMSE and max |error| are written to
calculator_validation.csv on every build; judge the surface on the 200-rep run.
A 10-replicate partial run from before the code freeze is kept only as
runs/stale-v1-10rep-DO-NOT-USE and feeds nothing downstream.

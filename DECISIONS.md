# Decisions

What the spec said, what was done, why. Dates are 2026.

## D1. S1 variance nesting (accepted 28 Oct)
Spec: patient ICC_p and eye ICC_e, default ICC_e = ICC_p/2. As shares of total
variance these exceed 1 at ICC_p = 0.9. ICC_e is therefore the eye share of the
within-patient variance: s2_eye = ICC_e(1 - ICC_p), s2_res = (1 - ICC_p)(1 - ICC_e).
Fellow eyes correlate at ICC_p, repeats of one eye at ICC_p + s2_eye. Components
are asserted to sum to 1 in code and in test 3b.

## D2. True AUC
True AUC is the oracle AUC of w'x on an independent k = 1 sample. Since every
feature has unit variance, AUC = Phi(beta/sqrt 2) and beta = sqrt(2) Phi^-1(AUC)
for every p and ICC: 0, 0.5449, 1.1902. Monte Carlo calibration per p reproduces
this (calibration.csv); the per-p split in the spec has no effect.

## D3. Test 3 (accepted 29 Oct)
Spec: at n = 200, ICC = 0, true AUC 0.80, every split within 0.80 +/- 0.02.
0.80 is the oracle AUC; learned models sit below it regardless of leakage, so
the test cannot pass as written. First fix compared splits with a model trained
on all patients; on v1 this failed in 4 LR p = 180 rows by -0.020 on every split,
a learning-curve effect (CV trains on 80% of patients). Final version: comparator
is the same model trained on a stratified random 80% of patients, scored on
2000 independent patients (scripts/test3_comparator.py). Each replicate is
regenerated from its seed and must reproduce the stored full-data AUC exactly
before the comparator is accepted. Tolerance stays 0.02. Result: max gap 0.0057.

## D4. CV AUC estimator (accepted 28 Oct)
Pooled out-of-fold AUC is biased below 0.5 under the null when fold class
balance varies, and patient-grouped folds vary more. At ICC = 0, n = 20 this
produced spurious optimism of +0.013 to +0.019 in S2/S3. Mean of per-fold AUCs
removed it. Fold-mean is primary; pooled is kept as *_pooled. On v1 the pooled
estimator would have added +0.003 (LR, kNN) and +0.008 (RF) at ICC = 0.
Test folds with one class are skipped and counted.

## D5. Degenerate draws
Patient labels are redrawn if either class has fewer than 2 patients
(probability ~4e-5 at n = 20). Count stored as redraws_total.

## D6. Calculator scope
Fitted per classifier x p on control, S1_r1_d0, S2_k4, S2_k8, S3: 300 cells each.
S1_r1_d0 is included because one row per eye with concordant labels is the same
as a k = 2 patient cluster. Other S1 variants are not a function of k_mean (eye
level adds a second clustering layer; d = 0.2 puts different labels in one
patient). Form: icc (1 - 1/k) poly2(icc, ln n, ln k, true_auc), zero at ICC = 0
and k = 1 by construction. Degree 3 was tried and was worse by leave-one-out.
true_auc is a keyword with default 0.65. p is categorical {20, 180}.
Held-out error on v1: RMSE 0.013-0.023, max 0.04-0.08, against MC SE ~0.003,
so the misfit is systematic. Report the calculator as accurate to about 0.02.

## D7. Test gates (accepted 29 Oct)
Tests 1 and 2 gate on |z| < 3 per cell (z = optimism / MC SE) with at most 1% of
cells failing; |optimism| < 0.01 is reported as a column, not gated. Test 0
requires 200 replicates per cell. On v1: test 1 failing cells 1/240 (LR), 0, 0;
test 2 1/120 (kNN), 0 (LR), 1/120 (RF). The kNN control cell (n 20, p 20,
ICC 0.9, AUC 0.8, z -4.05) was rerun for 200 more replicates (v1-ctrl-ext):
-0.0048 +/- 0.0049, z -0.98, combined z -3.54. Under the rule set before the
rerun this is inconclusive. The other 23 n = 20 control cells pooled give
+0.0019 +/- 0.0010 (kNN) and +0.0015 +/- 0.0012 (LR), so there is no systematic
splitter effect; the original cell is selection plus chance. The flagged RF
control cell (n 200, p 180, ICC 0.3, z +3.38) was not followed up.

## D8. Figures
Figure 1: true AUC 0.65 (flag --true-auc), S1 panel is r = 2, d = 0, bars are
95% MC intervals. Figure 2: split (a), colour = ICC, line style = classifier,
grey = patient split. Manuscript versions come from scripts/figures_v2.py;
mie27.figures output is superseded.

## D9. LR max_iter
1000 instead of the sklearn default 100 so lbfgs converges at p = 180. Other
settings are defaults.

## D10. Compute
Gadi was dropped; everything ran on one 250-core node via jobs/launch_local.sh.
RF cells are split into replicate chunks; chunked output is bit-identical to
unchunked. Cost: RF 1268 core-hours, LR 21.6, kNN 7.5.

## D11. Bugs fixed during v1
A leftover sandbox path in requirements.lock (fix lock file commit). Worker
pickling failed under `python -m` with more than one worker (fix multi-worker
entry point, b80d4a3); no cells were written by the failed attempts. nproc
honoured OMP_NUM_THREADS=1 and reported one core; fixed in the launcher.

## D12. S1 kNN non-monotone in ICC
At p = 180 kNN optimism in S1 is near its maximum by ICC 0.3 and falls by up to
0.028 from 0.6 to 0.9. A separate diagnostic (not v1) shows the patient's own
rows already fill the neighbour list; at high ICC the remaining slots fill with
rows of one other patient, whose shared label adds noise. At p = 20 it is graded.

## D13. Real-data check (realdemo-v1, 29 Oct)
PhysioNet/CinC 2019 set A, CC BY 4.0 (DATA.md). Design as specified by the
orchestrator; pipeline reuses mie27.cv.make_folds. Choices: imputation medians
fitted on the training fold; kNN standardised (v1 features already had unit
variance); ICC fed to the calculator is computed on the imputed matrix the
models see (0.25), not observed values only (0.70), since the latter would
predict optimism the models cannot realise on 80%-imputed features.
Result: observed optimism 0.03-0.26, all positive; calculator within its
stated error for LR (+0.02) and kNN (-0.05 to +0.01), under-predicts RF by
0.07-0.10. Trends with k and n match. Assumption violations are listed in
runs/realdemo-v1/limitations.md.

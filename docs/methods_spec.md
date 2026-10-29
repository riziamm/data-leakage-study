# Methods specification (mie27-unit-leakage, run_id v1, git b80d4a3)

Text below is written from the code as run; every value can be checked in
`src/mie27/generate.py`, `calibrate.py`, `cv.py` and `run.py`.

## Generative model (all scenarios)
Each of n patients has a label y_i ~ Bernoulli(0.5); a draw is repeated if either
class has fewer than 2 patients. Each row has p features
x = √ICC_p · u_i + (eye term, S1 only) + √(residual) · e_row + β · y_row · w,
with u_i, e_row ~ N(0, I_p) and w the fixed unit vector with weight 1/√5 on the
first 5 features and 0 elsewhere. The variance components of every feature sum to
1 in every scenario (asserted in code), so x | y ~ N(β·y·w, I_p) for every row.

## S3: rows per patient, k_i ~ 1 + NB(4, 1), capped at 50
k_i = min(1 + K_i, 50), where K_i is negative binomial with mean 4 and dispersion
(size) 1, i.e. geometric on {0, 1, 2, ...} with P(K = j) = 0.2 · 0.8^j
(numpy `negative_binomial(n = 1, p = 0.2)`). Hence E[k] = 5, Var(k) = 20
(SD ≈ 4.5); P(k = 1) = 0.20; the cap truncates with probability P(K ≥ 50) = 0.8^50 ≈ 1.4 × 10⁻⁵.
All rows of a patient carry the patient label y_i.

## S1: two eyes, r repeats, label discordance d
Each patient contributes 2 eyes × r rows, r ∈ {1, 2}. Each eye's label equals y_i,
flipped independently with probability d ∈ {0, 0.2}; with d = 0.2 the two eyes
disagree with probability 2d(1 − d) = 0.32. Rows carry their eye's label, and
the signal term uses the eye label. Variance is nested: patient share ICC_p, eye
share ICC_e of the within-patient variance (default ICC_e = ICC_p / 2), so
σ²_eye = ICC_e(1 − ICC_p) and σ²_residual = (1 − ICC_p)(1 − ICC_e). Correlation
between fellow eyes = ICC_p; between repeats of one eye = ICC_p + σ²_eye.
S2: k ∈ {4, 8} visits per patient. Control: k = 1, where ICC has no effect on the
data distribution.

## True-AUC sweep
The true AUC is the AUC of the oracle score w′x on an independent k = 1 sample.
Because w′x | y ~ N(β·y, 1), AUC = Φ(β/√2), so β = √2 · Φ⁻¹(AUC):
β = 0, 0.5449 and 1.1902 for true AUC 0.50, 0.65 and 0.80, identical for p = 20
and p = 180 and for every ICC. Monte Carlo calibration on 100,000 independent
rows per p reproduced each target within 0.001.

## Cross-validation scheme
Five folds, repeated three times with independent shuffles; the same folds are
used for every classifier within a replicate.
(a) Row-level: stratified K-fold on row labels (scikit-learn `StratifiedKFold`,
shuffled).
(b) Unit-level: stratified group K-fold with the eye as group (S1 only); outside
S1 the unit is the row and (b) is identical to (a).
(c) Patient-level: stratified group K-fold with the patient as group
(`StratifiedGroupKFold`, shuffled; implemented as a fold-for-fold replica of
scikit-learn 1.9.1, tested identical); the code asserts that no patient appears in
more than one test fold or in both training and test data of a fold.
Per repeat, AUC is computed within each test fold and averaged across folds;
folds whose test set contains one class only are skipped and counted. The
replicate AUC is the mean over the three repeats. Pooled out-of-fold AUC is kept
as a sensitivity analysis. Optimism = AUC(a) − AUC(c), averaged over 200
replicates per cell; Monte Carlo SE = SD across replicates / √200.

## Classifiers
Default settings, no tuning: L2 logistic regression (C = 1; features standardised
on the training fold; max_iter 1000 for convergence), kNN (k = 5, Euclidean,
unstandardised), random forest (200 trees).

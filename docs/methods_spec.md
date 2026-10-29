# Methods text (run v1, code b80d4a3)

Written from the code as run. Check against src/mie27/generate.py,
calibrate.py, cv.py, run.py and scripts/realdemo.py.

## Generative model

Each of n patients has label y_i ~ Bernoulli(0.5); a draw is repeated if either
class has fewer than 2 patients. Each row has p features

    x = sqrt(ICC_p) u_i + [eye term, S1 only] + sqrt(residual) e_row + beta y_row w

with u_i, e_row ~ N(0, I_p) and w a fixed unit vector with weight 1/sqrt(5) on
the first 5 features. Variance components sum to 1 in every scenario, so
x | y ~ N(beta y w, I_p) for every row.

## Scenarios

Control: one row per patient; ICC has no effect.

S1 (eyes): 2 eyes x r rows, r in {1, 2}. Each eye's label equals y_i flipped
with probability d in {0, 0.2}; with d = 0.2 the two eyes disagree with
probability 0.32. Rows carry their eye's label. Variance is nested: patient
share ICC_p, eye share ICC_e = ICC_p/2 of the within-patient variance, so
s2_eye = ICC_e(1 - ICC_p) and s2_res = (1 - ICC_p)(1 - ICC_e). Fellow eyes
correlate at ICC_p, repeats of one eye at ICC_p + s2_eye.

S2 (imaging): k in {4, 8} rows per patient, all with the patient label.

S3 (EHR): k_i = min(1 + K_i, 50), K_i negative binomial with mean 4 and
dispersion 1, i.e. P(K = j) = 0.2 x 0.8^j. E[k] = 5, SD 4.5, P(k = 1) = 0.20,
cap reached with probability 0.8^50.

## True AUC

Defined as the AUC of the oracle score w'x on an independent single-row sample.
w'x | y ~ N(beta y, 1), so AUC = Phi(beta / sqrt 2) and beta = sqrt(2) Phi^-1(AUC):
0, 0.5449, 1.1902 for 0.50, 0.65, 0.80, independent of p and ICC. Monte Carlo
calibration on 100,000 rows per p reproduces each target within 0.001.

## Cross-validation

Five folds, three repeats with independent shuffles, same folds for every
classifier within a replicate.
(a) row-level: StratifiedKFold on row labels;
(b) unit-level: StratifiedGroupKFold with eye as group (S1), identical to (a)
elsewhere;
(c) patient-level: StratifiedGroupKFold with patient as group (fold-for-fold
replica of scikit-learn 1.9.1, tested identical). Code asserts no patient is
in two test folds or in train and test of one fold.
AUC is computed per test fold and averaged over folds (single-class test folds
skipped and counted), then over the three repeats. Pooled out-of-fold AUC is a
sensitivity analysis. Optimism = AUC(a) - AUC(c), mean over 200 replicates;
MC SE = SD / sqrt(200).

## Classifiers

Defaults, no tuning. L2 logistic regression, C = 1, features standardised on the
training fold, max_iter 1000. kNN, k = 5, Euclidean, unstandardised (features
have unit variance). Random forest, 200 trees.

## Grid

n in {20, 30, 50, 100, 200}; ICC in {0, 0.3, 0.6, 0.9}; p in {20, 180}; true AUC
in {0.50, 0.65, 0.80}; 8 scenario variants; 200 replicates per cell; per-cell
seeds hashed from base seed 20271.

## Real-data check

PhysioNet/CinC 2019 training set A, 20,336 patients (DATA.md). Patient label =
any SepsisLabel = 1. Features = the 20 vitals/labs with least row missingness,
median-imputed and standardised on the training fold. Rows = first k hours,
k = 4 and 8; no patient had fewer than 8 rows. Subsamples of n = 50, 100, 200
patients, class-balanced, 100 per n. Same fold code, estimator and classifier
settings as the simulation. ICC per feature by one-way random-effects ANOVA on
the full cohort at the same k, on the imputed matrix (and on observed values
only, reported). Calculator input: n, k, mean imputed ICC, p = 20, observed
patient-split AUC as true_auc.

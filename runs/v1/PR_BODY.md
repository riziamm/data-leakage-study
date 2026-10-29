## Tests
============================= test session starts ==============================
platform linux -- Python 3.11.18, pytest-9.2.0, pluggy-1.7.0 -- /home/u5499379/projects/mie27-unit-leakage/.venv/bin/python
rootdir: /home/u5499379/projects/mie27-unit-leakage
configfile: pyproject.toml
collecting ... collected 77 items

tests/test_acceptance.py::test_0_full_replication PASSED                 [  1%]
tests/test_acceptance.py::test_1_icc0_no_optimism[lr] PASSED             [  2%]
tests/test_acceptance.py::test_1_icc0_no_optimism[knn] PASSED            [  3%]
tests/test_acceptance.py::test_1_icc0_no_optimism[rf] PASSED             [  5%]
tests/test_acceptance.py::test_2_control_no_optimism[lr] PASSED          [  6%]
tests/test_acceptance.py::test_2_control_no_optimism[knn] PASSED         [  7%]
tests/test_acceptance.py::test_2_control_no_optimism[rf] PASSED          [  9%]
tests/test_acceptance.py::test_3_splits_match_independent_auc PASSED     [ 10%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-control] PASSED [ 11%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-S1_r1_d0] PASSED [ 12%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-S1_r1_d0.2] PASSED [ 14%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-S1_r2_d0] PASSED [ 15%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-S1_r2_d0.2] PASSED [ 16%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-S2_k4] PASSED [ 18%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-S2_k8] PASSED [ 19%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.0-S3] PASSED [ 20%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-control] PASSED [ 22%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-S1_r1_d0] PASSED [ 23%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-S1_r1_d0.2] PASSED [ 24%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-S1_r2_d0] PASSED [ 25%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-S1_r2_d0.2] PASSED [ 27%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-S2_k4] PASSED [ 28%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-S2_k8] PASSED [ 29%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.3-S3] PASSED [ 31%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-control] PASSED [ 32%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-S1_r1_d0] PASSED [ 33%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-S1_r1_d0.2] PASSED [ 35%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-S1_r2_d0] PASSED [ 36%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-S1_r2_d0.2] PASSED [ 37%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-S2_k4] PASSED [ 38%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-S2_k8] PASSED [ 40%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.6-S3] PASSED [ 41%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-control] PASSED [ 42%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-S1_r1_d0] PASSED [ 44%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-S1_r1_d0.2] PASSED [ 45%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-S1_r2_d0] PASSED [ 46%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-S1_r2_d0.2] PASSED [ 48%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-S2_k4] PASSED [ 49%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-S2_k8] PASSED [ 50%]
tests/test_acceptance.py::test_3b_variance_components_sum_to_one[0.9-S3] PASSED [ 51%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-control] PASSED [ 53%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-S1_r1_d0] PASSED [ 54%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-S1_r1_d0.2] PASSED [ 55%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-S1_r2_d0] PASSED [ 57%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-S1_r2_d0.2] PASSED [ 58%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-S2_k4] PASSED [ 59%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-S2_k8] PASSED [ 61%]
tests/test_acceptance.py::test_4_patient_split_disjoint[20-S3] PASSED    [ 62%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-control] PASSED [ 63%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-S1_r1_d0] PASSED [ 64%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-S1_r1_d0.2] PASSED [ 66%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-S1_r2_d0] PASSED [ 67%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-S1_r2_d0.2] PASSED [ 68%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-S2_k4] PASSED [ 70%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-S2_k8] PASSED [ 71%]
tests/test_acceptance.py::test_4_patient_split_disjoint[200-S3] PASSED   [ 72%]
tests/test_acceptance.py::test_4b_guard_fires_on_leaky_folds PASSED      [ 74%]
tests/test_acceptance.py::test_5_calibration[0.5-20] PASSED              [ 75%]
tests/test_acceptance.py::test_5_calibration[0.5-180] PASSED             [ 76%]
tests/test_acceptance.py::test_5_calibration[0.65-20] PASSED             [ 77%]
tests/test_acceptance.py::test_5_calibration[0.65-180] PASSED            [ 79%]
tests/test_acceptance.py::test_5_calibration[0.8-20] PASSED              [ 80%]
tests/test_acceptance.py::test_5_calibration[0.8-180] PASSED             [ 81%]
tests/test_fast_equivalence.py::test_auc_matches_sklearn_including_ties PASSED [ 83%]
tests/test_fast_equivalence.py::test_knn_matches_sklearn[S2_k8-50-180] PASSED [ 84%]
tests/test_fast_equivalence.py::test_knn_matches_sklearn[S3-30-20] PASSED [ 85%]
tests/test_fast_equivalence.py::test_knn_matches_sklearn[S1_r2_d0.2-20-180] PASSED [ 87%]
tests/test_fast_equivalence.py::test_lr_matches_sklearn_pipeline[S2_k8-50-180] PASSED [ 88%]
tests/test_fast_equivalence.py::test_lr_matches_sklearn_pipeline[S3-30-20] PASSED [ 89%]
tests/test_fast_equivalence.py::test_lr_matches_sklearn_pipeline[control-20-180] PASSED [ 90%]
tests/test_fast_equivalence.py::test_stratified_group_kfold_identical_to_sklearn[control] PASSED [ 92%]
tests/test_fast_equivalence.py::test_stratified_group_kfold_identical_to_sklearn[S1_r1_d0.2] PASSED [ 93%]
tests/test_fast_equivalence.py::test_stratified_group_kfold_identical_to_sklearn[S1_r2_d0.2] PASSED [ 94%]
tests/test_fast_equivalence.py::test_stratified_group_kfold_identical_to_sklearn[S2_k8] PASSED [ 96%]
tests/test_fast_equivalence.py::test_stratified_group_kfold_identical_to_sklearn[S3] PASSED [ 97%]
tests/test_reproducibility.py::test_extending_replicates_equals_running_at_once PASSED [ 98%]
tests/test_reproducibility.py::test_models_run_separately_see_same_data PASSED [100%]

============================= 77 passed in 26.32s ==============================

## Calculator
in-sample / LOO fit error:
         n_cells     rmse  max_abs  loo_rmse  mc_se_mean
knn|20     300.0  0.01465  0.06504   0.01581     0.00328
knn|180    300.0  0.01302  0.04278   0.01402     0.00345
lr|20      300.0  0.01679  0.04961   0.01824     0.00319
lr|180     300.0  0.01996  0.07313   0.02179     0.00439
rf|20      300.0  0.01740  0.05531   0.01878     0.00382
rf|180     300.0  0.02287  0.07513   0.02491     0.00453

held-out validation (20% of cells):
model   p  n_heldout  heldout_rmse  heldout_max_abs  heldout_mc_se_mean
  knn  20         69      0.013223         0.048052            0.002690
  knn 180         69      0.014481         0.040549            0.002916
   lr  20         69      0.015499         0.042927            0.002558
   lr 180         69      0.021092         0.063664            0.003614
   rf  20         69      0.015787         0.044326            0.003096
   rf 180         69      0.023443         0.081981            0.003776

## Calibration
p,target_auc,beta,beta_mc,achieved_auc_check,n_test
20,0.5,0.0,0.0,0.49957424550870133,100000
20,0.65,0.5449254294535087,0.541778564453125,0.6498559777140167,100000
20,0.8,1.1902321628999901,1.185699462890625,0.8001126061361108,100000
180,0.5,0.0,0.0,0.499770136059709,100000
180,0.65,0.5449254294535087,0.553253173828125,0.6497489832282779,100000
180,0.8,1.1902321628999901,1.198638916015625,0.7999457658600106,100000

## Code
uncommitted files: 8

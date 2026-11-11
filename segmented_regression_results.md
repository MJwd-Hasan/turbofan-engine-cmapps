# FD001 Q0.5 Segmented Regression Results

## Executive summary

The segmented regression experiments now use one fixed knot: the median
(Q0.5) of training cycle-row values, **cycle 104**. Rows at cycle 104 are
assigned to the early segment; rows after 104 are assigned to the late
segment. The knot is calculated from training data and is not selected using
test outcomes.

The best test MAE among the reported segmented regressors is **19.128
cycles**, from the fixed-Q0.5 PCA-XGBoost pipeline. A second PCA-XGBoost
pipeline has the lowest RMSE (**26.262**) and highest R-squared (**0.601**).
The SHAP-screened XGBoost, segmented LightGBM, and segmented LSTM provide
additional Q0.5 comparisons. These rankings vary by metric and are descriptive
results on the supplied 100-engine FD001 test set, not evidence of external
generalization or a biological change point.

| Model | Test C-index | Test MAE | Test RMSE | Test R-squared |
|---|---:|---:|---:|---:|
| SHAP-screened, binned-sensor XGBoost | 0.830 | 20.108 | 26.310 | Not reported |
| Fixed-Q0.5 PCA-XGBoost | **0.836** | **19.128** | 26.404 | 0.596 |
| Grouped-CV-tuned Q0.5 PCA-XGBoost | 0.830 | 19.844 | **26.262** | **0.601** |
| Fixed-parameter Q0.5 LightGBM | **0.836** | 20.736 | 27.880 | 0.550 |
| SHAP-informed Q0.5 LSTM | 0.826 | 22.349 | 29.437 | 0.498 |

## Data, target, and segmentation

- **Training data:** `train_FD001.txt`, 20,631 cycle rows from 100 engines.
- **Test data:** `test_FD001.txt`, partial trajectories for 100 engines.
- **Test outcomes:** `RUL_FD001.txt`, one remaining-life label per test engine.
- **Training response:** for engine \(i\), let \(T_i\) be its final training
  cycle. At cycle \(t\), the response is \(T_i-t\).
- **Test evaluation:** one prediction per engine, made from its last observed
  test row and compared with the supplied RUL label.
- **Fixed knot:** `train_rows["cycle"].quantile(0.5)` is cycle 104.
- **Segment assignment:** early is cycle \(\leq104\), late is cycle \(>104\).
  Test engines are assigned by their last observed cycle.

The median value is cycle 104; because cycle is discrete and many rows share
the knot, 10,400 training rows (about 50.4%) fall at or below it and 10,231
fall above it. The test set has 31 early-segment engines and 69 late-segment
engines. This is a fixed median split, not a search for an optimal change
point.

## Methodology and validation

The training observations are repeated measurements within engines, not
independent samples. Where validation is reported, `GroupKFold` keeps each
engine entirely in one fold. Pseudo-censor validation selects intermediate
cycles from held-out training engines and evaluates RUL at those cutoffs.
Preprocessing and segment models are fit using training-fold engines only.
Apparent training fit is reported separately and is optimistic.

The knot-validation cell evaluates the fixed Q0.5 split with three
engine-grouped folds; it does not compare or select candidate knots. Across
the folds it evaluates 600 pseudo-censor rows, with fold MAEs **21.048**,
**25.342**, and **24.458** cycles. Pooled MAE is **23.634** and pooled RMSE is
**32.752** cycles.

### Model designs

1. **SHAP-screened XGBoost:** separate early/late models with cycle, five
   numeric SHAP-selected sensors, and binned `sensor_7` and `sensor_13`.
   Fifty Optuna trials and 50 Adaptive Greedy Search evaluations were compared
   per segment; the selected method was Optuna early and AGS late. The
   unadjusted sensor screening threshold was \(p<0.10\); neither additional
   sensor passed 5% Benjamini-Hochberg FDR correction.
2. **Fixed-Q0.5 PCA-XGBoost:** separate PCA preprocessing and XGBoost models
   use cycle plus varying settings/sensors. Each segment uses 30 AGS
   evaluations. The models retain 12 early and 9 late principal components
   for at least 90% explained training variance.
3. **Grouped-CV-tuned Q0.5 PCA-XGBoost:** a second PCA-XGBoost pipeline also
   fits separate segment models and uses 30 AGS evaluations per segment. Its
   fold and pseudo-censor construction differ from the fixed-threshold
   pipeline, so its metrics are not a controlled comparison with that model.
4. **Fixed-parameter Q0.5 LightGBM:** separate models use cycle and all
   varying settings/sensors. Each has 300 estimators, learning rate 0.05,
   31 leaves, row and feature subsampling of 0.85, and L2 regularization 1.0.
   Three-fold engine-grouped pseudo-censor validation is used; there is no
   hyperparameter search. Reported LightGBM importances are gain-based, not
   SHAP.
5. **SHAP-informed Q0.5 LSTM:** segment-specific 15-cycle sequences, with
   eight sensor inputs selected per segment by a tree-surrogate SHAP screen.
   Eighty engines are used for training and 20 engine-disjoint engines for
   validation; early stopping selects the epoch separately for each segment.
   Target RUL is standardized during training and predictions are returned to
   cycles.

The earlier single-model booster benchmarks remain in the notebook as
unsegmented references; the Q0.5 experiments above provide segmented
counterparts for the regression model families. The survival estimators are
handled separately below because their event-time estimands are not segmented
regressions.

## Results by model

### 1. SHAP-screened XGBoost

| Segment | Training rows | Apparent train MAE | Grouped pseudo-censor CV MAE | Grouped CV RMSE |
|---|---:|---:|---:|---:|
| Early, cycle \(\leq104\) | 10,400 | 26.461 | 30.939 | 39.580 |
| Late, cycle \(>104\) | 10,231 | 17.428 | 19.236 | 26.591 |

| Evaluation cohort | C-index | Dynamic AUC | MAE | RMSE | IAE | ISE | Median predicted RUL | Median actual RUL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grouped pseudo-censor validation | 0.804 | 0.847 | 24.941 | 33.557 | 3.531 | 0.114 | 101.481 | 90.000 |
| FD001 test set | 0.830 | 0.924 | 20.108 | 26.310 | 6.421 | 0.465 | 81.703 | 86.000 |

Test engines per segment are 31 early and 69 late. AIC/BIC are not defined for
these separate XGBoost prediction fits.

Top mean absolute SHAP attributions for the original/binned features:

| Rank | Early feature | Mean \(|SHAP|\) | Late feature | Mean \(|SHAP|\) |
|---:|---|---:|---|---:|
| 1 | cycle | 18.2667 | sensor_14 | 9.5246 |
| 2 | sensor_14 | 5.2856 | sensor_12 | 8.7029 |
| 3 | sensor_8 | 4.4694 | sensor_8 | 4.5802 |
| 4 | sensor_12 | 4.1339 | sensor_2 | 3.9526 |
| 5 | sensor_2 | 3.0765 | sensor_3 | 3.3491 |

### 2. Fixed-Q0.5 PCA-XGBoost

| Segment | Train rows | PCA components | Apparent train MAE | Grouped CV rows | Grouped CV MAE | Grouped CV RMSE | Grouped CV R-squared |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early, cycle \(\leq104\) | 10,400 | 12 | 31.995 | 400 | 33.354 | 44.053 | 0.326 |
| Late, cycle \(>104\) | 10,231 | 9 | 15.617 | 400 | 12.705 | 20.053 | 0.663 |

| Test segment | Engines | MAE | RMSE | R-squared |
|---|---:|---:|---:|---:|
| Early, cycle \(\leq104\) | 31 | 31.593 | 37.357 | -2.785 |
| Late, cycle \(>104\) | 69 | 13.528 | 19.580 | 0.746 |

| Overall test metric | Result |
|---|---:|
| C-index | 0.836 |
| MAE | 19.128 cycles |
| RMSE | 26.404 cycles |
| R-squared | 0.596 |
| Median predicted RUL | 79.206 cycles |
| Median actual RUL | 86 cycles |

Best AGS parameters:

- **Early:** 140 estimators, depth 2, learning rate 0.10, minimum child weight
  10, subsample 0.85, column subsample 0.85, L2 regularization 20.0, gamma
  0.25.
- **Late:** 180 estimators, depth 5, learning rate 0.025, minimum child weight
  1, subsample 0.65, column subsample 1.0, L2 regularization 0.5, gamma 3.0.

Residual diagnostics (descriptive):

| Evaluation data | Shapiro p-value | Wilcoxon median-zero p-value | Spearman \(|error|\)-prediction rho | Spearman p-value |
|---|---:|---:|---:|---:|
| Grouped pseudo-censor OOF | 2.481e-21 | 7.193e-11 | 0.5782 | 1.485e-72 |
| Test engines | 0.05576 | 0.001579 | 0.5889 | 1.163e-10 |

The early test-segment R-squared is strongly negative, while the late-segment
score is positive. Aggregate test metrics therefore conceal a substantial
early-regime weakness.

Top mean absolute Tree SHAP values are cycle (21.8795), `pca__pca0` (12.7419),
and `pca__pca1` (5.9255) early; `pca__pca0` (28.3560), `pca__pca1` (11.0057),
and cycle (1.8835) late. These PCA-component attributions do not map
one-to-one to original sensors.

### 3. Grouped-CV-tuned Q0.5 PCA-XGBoost

| Segment | Train rows | PCA components | Best AGS grouped MAE | OOF rows | OOF MAE | OOF RMSE | OOF R-squared |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early, cycle \(\leq104\) | 10,400 | 12 | 31.450 | 275 | 31.486 | 39.702 | 0.190 |
| Late, cycle \(>104\) | 10,231 | 9 | 16.871 | 325 | 16.876 | 25.267 | 0.651 |

| Overall test metric | Result |
|---|---:|
| C-index | 0.830 |
| MAE | 19.844 cycles |
| RMSE | 26.262 cycles |
| R-squared | 0.601 |
| Median predicted RUL | 81.041 cycles |
| Median actual RUL | 86 cycles |

There were 31 early and 69 late test engines. Apparent training MAE/RMSE were
32.269/42.958 early and 17.640/25.611 late; these are in-sample fit
diagnostics, not validation estimates. Best AGS parameters:

- **Early:** 220 estimators, depth 2, learning rate 0.025, minimum child weight
  1, subsample 0.75, column subsample 1.0, L2 regularization 10.0, gamma 1.5.
- **Late:** 140 estimators, depth 2, learning rate 0.06, minimum child weight
  4, subsample 1.0, column subsample 0.85, L2 regularization 0.1, gamma 1.5.

| Evaluation data | Shapiro p-value | Wilcoxon median-zero p-value | Spearman \(|error|\)-prediction rho |
|---|---:|---:|---:|
| Grouped pseudo-censor OOF | 1.69e-14 | 8.284e-14 | 0.4982 |
| Test engines | 0.3421 | 0.0003826 | 0.5707 |

### 4. Fixed-parameter Q0.5 LightGBM

| Segment | Pseudo-censor validation rows | Validation MAE | Validation RMSE | Validation R-squared |
|---|---:|---:|---:|---:|
| Early, cycle \(\leq104\) | 167 | 34.571 | 42.043 | -0.187 |
| Late, cycle \(>104\) | 233 | 18.219 | 27.999 | 0.529 |

| Test segment | Engines | MAE | RMSE | R-squared |
|---|---:|---:|---:|---:|
| Early, cycle \(\leq104\) | 31 | 34.440 | 40.109 | -3.363 |
| Late, cycle \(>104\) | 69 | 14.579 | 20.093 | 0.733 |

| Overall test metric | Result |
|---|---:|
| C-index | 0.836 |
| MAE | 20.736 cycles |
| RMSE | 27.880 cycles |
| R-squared | 0.550 |
| Median predicted RUL | 79.208 cycles |
| Median actual RUL | 86 cycles |

Test residual diagnostics are Shapiro p=0.09165, Wilcoxon median-zero
p=0.001, and Spearman correlation between absolute error and prediction
\(\rho=0.6571\). These are descriptive tests, not LightGBM assumptions.

Top gain importances (not SHAP):

| Rank | Early feature | Early gain | Late feature | Late gain |
|---:|---|---:|---|---:|
| 1 | cycle | 92,361,801.484 | sensor_11 | 72,262,855.468 |
| 2 | sensor_11 | 18,523,064.596 | sensor_4 | 29,689,720.418 |
| 3 | sensor_12 | 10,565,659.567 | sensor_9 | 19,512,126.515 |
| 4 | sensor_14 | 9,649,813.792 | sensor_12 | 16,454,406.319 |
| 5 | sensor_4 | 9,552,901.909 | sensor_14 | 10,325,408.611 |
| 6 | sensor_9 | 9,017,698.696 | sensor_7 | 9,424,791.154 |
| 7 | sensor_15 | 7,245,519.830 | sensor_15 | 9,399,945.678 |
| 8 | sensor_7 | 7,073,390.676 | sensor_21 | 4,120,117.135 |
| 9 | sensor_21 | 6,467,498.810 | sensor_2 | 4,036,315.940 |
| 10 | sensor_3 | 5,837,791.926 | sensor_8 | 3,829,661.264 |

### 5. SHAP-informed Q0.5 LSTM

| Segment | Training rows | Best epoch | Apparent train MAE | Validation rows / engines | Validation MAE | Validation RMSE | Validation R-squared |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early, cycle \(\leq104\) | 10,400 | 5 | 32.526 | 33 / 20 | 28.530 | 34.509 | 0.368 |
| Late, cycle \(>104\) | 10,231 | 21 | 14.869 | 47 / 20 | 10.207 | 14.780 | 0.841 |

| Overall test metric | Result |
|---|---:|
| C-index | 0.826 |
| MAE | 22.349 cycles |
| RMSE | 29.437 cycles |
| R-squared | 0.498 |
| Median predicted RUL | 74.462 cycles |
| Median actual RUL | 86 cycles |

There were 31 early and 69 late test engines. Test residual Shapiro p=0.3768;
held-out validation residual Shapiro p=0.003584. Normal residuals are not
required for an LSTM.

The segment-specific SHAP-screened sensor inputs and mean absolute surrogate
Tree SHAP values were:

| Rank | Early sensor | Early mean \(|SHAP|\) | Late sensor | Late mean \(|SHAP|\) |
|---:|---|---:|---|---:|
| 1 | `sensor_14` | 6.6709 | `sensor_11` | 9.3357 |
| 2 | `sensor_11` | 5.9450 | `sensor_9` | 6.5954 |
| 3 | `sensor_9` | 5.0966 | `sensor_4` | 6.4218 |
| 4 | `sensor_4` | 2.9257 | `sensor_14` | 4.9421 |
| 5 | `sensor_12` | 2.7188 | `sensor_12` | 3.9809 |
| 6 | `sensor_21` | 2.5460 | `sensor_7` | 3.6154 |
| 7 | `sensor_15` | 2.3799 | `sensor_15` | 3.0890 |
| 8 | `sensor_2` | 1.8434 | `sensor_20` | 2.4037 |

These are selections from tree-surrogate SHAP rankings, not direct LSTM
explanations or causal effects.

## Interpretation and limitations

- **Metric-dependent ranking:** fixed-Q0.5 PCA-XGBoost has the lowest test
  MAE; the grouped-CV-tuned PCA-XGBoost has the lowest RMSE and highest
  R-squared. LightGBM ties the highest C-index. No model is best on every
  reported metric.
- **Early-regime difficulty:** the test set has only 31 early-segment engines.
  The fixed PCA-XGBoost and LightGBM have negative early-segment R-squared,
  meaning their predictions are worse than using the early-segment test mean
  as a predictor. Segment-specific test estimates should be interpreted with
  caution.
- **Fixed knot, not a discovered change point:** Q0.5 is a design choice from
  the training cycle-row distribution. Validation scores do not prove a
  biological transition at cycle 104.
- **No theorem-based guarantee:** the implementation applies a fixed-knot
  segmented-regression modeling procedure; it does not claim a separate
  statistical theorem guarantees better prediction or a true change point.
- **Engine dependence:** cycle rows are repeated measurements. Grouped
  validation avoids engine overlap, but pseudo-censor cutoffs from one engine
  are still correlated observations; row counts are not independent sample
  sizes.
- **Test-set reuse:** the same 100 test engines have been used across multiple
  experiments. Reported test metrics are descriptive and may be optimistic
  after repeated model development.
- **Survival models:** the log-logistic, Cox, Kaplan-Meier, Nelson-Aalen, and
  random-survival-forest analyses remain duration-based. The cycle-knot
  regression procedure is not a direct substitute for their hazard or
  survival-function estimands, so they were not segmented using this knot.
- **Interpretability:** PCA-XGBoost SHAP refers to principal components;
  LightGBM gain importance is not SHAP; LSTM inputs are screened using
  surrogate-tree SHAP. None establishes causal sensor effects.
- **AIC/BIC:** these likelihood criteria are not appropriate for the reported
  XGBoost, LightGBM, or LSTM prediction fits.

## Figures and notebook

The executed analysis and cell outputs are in [`fd001_trial.ipynb`](./fd001_trial.ipynb).
Relevant plots include:

- [`knot_optimization.png`](./images/knot_optimization.png): fixed-Q0.5
  grouped pseudo-censor validation.
- [`optimized_knot_xgb_diagnostics.png`](./images/optimized_knot_xgb_diagnostics.png):
  PCA-XGBoost predictions and residuals.
- [`optimized_knot_xgb_survival.png`](./images/optimized_knot_xgb_survival.png):
  empirical predicted-versus-actual test RUL distributions.
- [`optimized_knot_xgb_ags_history.png`](./images/optimized_knot_xgb_ags_history.png):
  segment-specific tuning histories.
- [`optimized_knot_xgb_shap_early.png`](./images/optimized_knot_xgb_shap_early.png)
  and [`optimized_knot_xgb_shap_late.png`](./images/optimized_knot_xgb_shap_late.png):
  PCA-component Tree SHAP summaries.
- [`segmented_lstm_training.png`](./images/segmented_lstm_training.png),
  [`segmented_lstm_survival.png`](./images/segmented_lstm_survival.png), and
  [`segmented_lstm_diagnostics.png`](./images/segmented_lstm_diagnostics.png):
  Q0.5 LSTM training, test RUL distributions, and diagnostics.

The survival-curve plots compare empirical test RUL distributions; they are
not individual engine survival forecasts.

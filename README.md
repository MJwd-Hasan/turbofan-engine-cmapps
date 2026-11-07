# NASA CMAPSS FD001 — Turbofan Engine Remaining Useful Life

An exploratory and predictive analysis of turbofan engine degradation using the NASA C-MAPSS FD001 dataset. The notebook compares survival analysis, tree-based survival models, sensor reduction and clustering, gradient-boosting regressors, hyperparameter search, and SHAP-based explanations.

> **Headline result:** The final segmented XGBoost experiment reports test MAE **20.11 cycles**, C-index **0.830**, and cumulative/dynamic AUC **0.924** on the 100 FD001 test engines. This is the lowest test MAE reported in the notebook, but it is a later, differently specified experiment—not a controlled head-to-head win over every earlier model.

## Contents
- Project overview
- Dataset
- Notebook workflow
- Methodology
- Model outcomes
- SHAP analysis
- Interpretation and limitations
- Running the notebook

## Project overview

The objective is to estimate the **remaining useful life (RUL)** of a turbofan engine from operating-cycle history and sensor measurements. The notebook treats degradation as both a time-to-failure/survival-analysis problem and a remaining-life regression problem. It also investigates sensor selection, dimensionality reduction, clustering, hyperparameter optimization, and model interpretability.

The complete analysis, executed outputs, tables, plots, and discussion are in [findings.ipynb](findings.ipynb).

## Dataset

This project uses the **FD001** subset of the NASA C-MAPSS turbofan degradation dataset.

| File | Role | Observations |
|---|---|---:|
| train_FD001.txt | Complete run-to-failure training trajectories | 20,631 engine-cycle rows from 100 engines |
| test_FD001.txt | Partial trajectories for test engines | 13,096 engine-cycle rows from 100 engines |
| RUL_FD001.txt | Actual RUL for each test engine | 100 outcomes |

Each row has 26 columns: engine ID, cycle, three operational settings, and 21 sensor measurements. The inspected files contain no missing values. Several settings and sensors are constant in FD001 and are excluded from relevant variable-feature workflows.

**Important:** a row is an engine-cycle observation, not an independent engine. The 20,631 training rows are repeated measurements from 100 engines.

## Notebook workflow

1. Data inspection: dimensions, types, missingness, descriptive statistics, and Pearson-correlation heatmap.
2. Parametric survival analysis: log-logistic accelerated-failure-time model in the proportional-odds parameterization.
3. Semi-parametric survival analysis: Cox proportional-hazards regression and PH diagnostics.
4. Non-parametric survival analysis: Kaplan–Meier and Nelson–Aalen estimates.
5. Survival prediction: Kaplan–Meier, Cox PH, and random survival forest (RSF).
6. RSF explanation: SHAP values for a tree-based surrogate of the RSF risk score.
7. Sensor representation experiments: hierarchical correlation clustering, PCA + K-means, and full-covariance Gaussian mixture model (GMM).
8. Remaining-life regression: XGBoost and LightGBM.
9. Hyperparameter tuning: Optuna versus Adaptive Greedy Search (AGS), using engine-grouped pseudo-censor validation.
10. SHAP-guided feature engineering: continuous high-ranking sensors plus quantile-binned sensors.
11. Segmented regression: separate early- and late-cycle XGBoost models split at the median training cycle.

## Methodology

### Failure time and RUL target

For each training engine, the final observed cycle is treated as its failure cycle, \(T_i\). For a training observation at cycle \(t\):

\[
\mathrm{RUL}_{i,t}=T_i-t
\]

For test evaluation, the notebook uses each test engine's **last observed row** and compares its predicted RUL with the corresponding value in RUL_FD001.txt.

The initial survival-model experiments use one cycle-1 sensor record per engine and condition predicted survival on the test engine's observed age. The boosted regressors use cycle-level training rows and last-observed test rows.

### Survival models

- **Log-logistic model:** parametric model fitted to one baseline record per training engine. In the notebook's proportional-odds parameterization, positive alpha coefficients indicate longer survival.
- **Cox PH:** semi-parametric proportional-hazards model with ridge penalty 0.1. Hazard-ratio interpretations depend on the proportional-hazards assumption.
- **Kaplan–Meier:** non-parametric survival-function estimate.
- **Nelson–Aalen:** non-parametric cumulative-hazard estimate.
- **Random survival forest:** non-parametric ensemble survival model. Its SHAP explanation is indirect and uses a tree-based surrogate.

The training cohort has 100 observed failure events and a Kaplan–Meier median failure time of **199 cycles**.

### Sensor reduction and clustering

Three RSF feature representations are compared with an all-sensor baseline:

- **Hierarchical correlation clustering:** sensor distance is \(1-|r|\), where \(r\) is Pearson correlation. A training-only threshold is selected and one representative is retained from each cluster.
- **PCA + K-means:** PCA is fitted to standardized training sensors; the smallest number of components reaching at least 90% cumulative variance is retained. K-means cluster count is selected using training silhouette scores. A permutation test compares the selected silhouette with a shuffled-marginal null.
- **Full-covariance GMM:** fitted to standardized sensors without PCA; component count is selected by training BIC.

Feature selection, scaling, and clustering decisions in these workflows are learned from training data.

### Remaining-life regression and validation

XGBoost and LightGBM are trained on cycle-level RUL targets. Test predictions use each engine's final observed sensor row.

The tuning workflows use **GroupKFold** so that rows from one engine cannot appear on both sides of a fold. Validation observations are selected at pseudo-censoring cutoffs within held-out engines, mimicking RUL prediction before failure. Three grouped folds are used in the tuning experiments. The initial booster benchmark also uses five-fold grouped out-of-fold residuals to construct empirical residual distributions for survival plots.

- **Optuna:** 30 trials in the initial XGBoost/LightGBM comparison; 50 trials in later XGBoost experiments.
- **Adaptive Greedy Search:** 30 evaluations initially; 50 evaluations in later experiments. The notebook supplies precomputed engine-disjoint validation splits because the AGS interface used does not accept a groups argument.

Grouped pseudo-censor CV MAE is a validation estimate. Apparent in-sample training metrics are shown separately and are not treated as generalization estimates.

### SHAP-guided feature engineering and segmentation

The five numeric sensors selected from the RSF surrogate SHAP ranking are sensor_12, sensor_2, sensor_8, sensor_3, and sensor_14.

The feature-engineering experiment keeps these numeric sensors continuous and quantile-bins other varying sensors. Kruskal–Wallis tests assess binned sensor groups against engine failure cycle \(T\), using one baseline record per engine. Benjamini–Hochberg adjusted p-values are also reported.

The final experiment uses the unadjusted \(p<0.10\) threshold to select sensor_7 and sensor_13 as additional categorical predictors. It splits training rows at the median cycle, **104**, and fits separate early- and late-cycle XGBoost models. Quantile boundaries and one-hot encoding are learned within training folds.

## Model outcomes

All test-set results below refer to the supplied **100 FD001 test engines**. Model groups use different feature representations and prediction constructions; comparisons across sections are informative but not all strictly controlled.

### 1. Survival-model comparison

| Model | C-index | Cumulative/dynamic AUC | MAE (cycles) | IAE | ISE |
|---|---:|---:|---:|---:|---:|
| Kaplan–Meier | 0.7175 | 0.8166 | 31.2933 | 8.0390 | 0.7973 |
| Cox PH | 0.7100 | 0.7997 | 32.3554 | 8.1555 | 0.8176 |
| Random survival forest | 0.7159 | 0.7973 | 31.9272 | 8.5594 | 0.9416 |

Kaplan–Meier is slightly better than Cox PH and RSF on C-index, dynamic AUC, and MAE in this initial comparison. Cox PH reports partial AIC **740.42** and BIC **779.50**. AIC/BIC are not applicable to the KM and RSF estimates here.

The separate log-logistic model reports AIC **1093.95** and C-index **0.6486** on its engine-level fit. Its likelihood-based AIC should not be compared directly with Cox partial AIC.

### 2. RSF sensor-representation comparison

| RSF representation | C-index | Dynamic AUC | MAE (cycles) | IAE | ISE |
|---|---:|---:|---:|---:|---:|
| All sensors (baseline) | 0.7159 | 0.7973 | 31.9272 | 8.5594 | 0.9416 |
| Correlation-cluster representatives | **0.7239** | 0.8090 | 31.2478 | 8.3627 | 0.9171 |
| PCA + K-means | 0.7147 | 0.8024 | **31.1089** | 8.3710 | 0.8646 |
| Full-covariance GMM | 0.7148 | **0.8145** | 31.8103 | **8.3051** | **0.8328** |

Additional diagnostics:

- Hierarchical clustering retained **6 of 15** candidate sensors: sensor_9, sensor_11, sensor_2, sensor_17, sensor_3, and sensor_6. Selected distance threshold: **0.487**; silhouette: **0.144**.
- PCA suitability: KMO **0.928**; Bartlett's test \(\chi^2(105)=815.36\), \(p=2.45\times10^{-110}\). Ten components retained **91.7%** of variance.
- K-means selected **2 clusters**, silhouette **0.342**. The 99-permutation test returned \(p=0.140\), so the cluster structure was not clearly distinguishable from the shuffled-marginal null used in the notebook.
- GMM selected **2 components** by BIC, converged in 21 iterations, and had silhouette **0.181**. Component sizes: 81 and 19 engines.

The sensor-reduction variants show modest and mixed changes, not a consistent improvement across every metric.

### 3. Initial XGBoost and LightGBM benchmark

The boosters use final-cycle sensor measurements. Earlier RSF rows are references, but those RSF models use cycle-1 covariates and condition survival on test-engine age; this is not a perfectly controlled model-only comparison.

| Model | C-index | Dynamic AUC | MAE (cycles) |
|---|---:|---:|---:|
| Baseline RSF (all sensors) | 0.7159 | 0.7973 | 31.9272 |
| Correlation-representative RSF | 0.7239 | 0.8090 | 31.2478 |
| PCA + K-means RSF | 0.7147 | 0.8024 | 31.1089 |
| Full-covariance GMM RSF | 0.7148 | 0.8145 | 31.8103 |
| XGBoost | **0.8345** | **0.9264** | **23.0395** |
| LightGBM | 0.8336 | 0.9244 | 23.4507 |

Both boosted-tree models have higher reported C-index and dynamic AUC and lower MAE than the RSF variants in this benchmark. Their feature timing and prediction setup differ from the earlier RSF analysis.

### 4. Hyperparameter-search comparison

Initial tuning uses 30 Optuna trials or 30 AGS evaluations per model.

| Model | Search method | Best grouped CV MAE |
|---|---|---:|
| XGBoost | Optuna (30 trials) | 27.514 |
| XGBoost | Adaptive Greedy Search (30 evaluations) | **27.452** |
| LightGBM | Optuna (30 trials) | **27.613** |
| LightGBM | Adaptive Greedy Search (30 evaluations) | 27.871 |

Test-set results:

| Model | Median predicted RUL | Actual median RUL | MAE (cycles) | RMSE | C-index |
|---|---:|---:|---:|---:|---:|
| XGBoost + Optuna | 107.76 | 86.00 | 24.58 | 33.58 | 0.83 |
| XGBoost + AGS | 100.01 | 86.00 | 25.05 | 35.64 | 0.82 |
| LightGBM + Optuna | 102.12 | 86.00 | **23.70** | **32.31** | **0.84** |
| LightGBM + AGS | **96.20** | 86.00 | 24.52 | 34.37 | 0.83 |

LightGBM + Optuna has the lowest MAE and RMSE in this tuned comparison. LightGBM + AGS has a predicted median closest to the actual test median of 86 cycles. These are different summaries of error.

### 5. SHAP-guided, binned-sensor XGBoost

This experiment compares default XGBoost with 50-trial Optuna and 50-evaluation AGS tuning.

| Model | C-index | Dynamic AUC | MAE (cycles) | IAE | ISE | Median predicted RUL |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost default | 0.838 | 0.928 | 22.835 | 9.439 | 1.180 | 106.239 |
| XGBoost + Optuna (50 trials) | **0.844** | **0.933** | **22.230** | **8.662** | **0.998** | 103.772 |
| XGBoost + AGS (50 evaluations) | 0.838 | 0.930 | 22.905 | 9.990 | 1.257 | 108.030 |

Optuna leads on all five displayed metrics in this experiment. Its grouped pseudo-censor CV MAE is **27.392** cycles, compared with **27.587** for AGS.

Kruskal–Wallis tests on engine-level baseline sensor bins:

| Sensor | Kruskal–Wallis p-value | BH-adjusted p-value |
|---|---:|---:|
| sensor_7 | 0.09905 | 0.4457 |
| sensor_13 | 0.08321 | 0.4457 |

These sensors meet the exploratory unadjusted \(p<0.10\) threshold, but **neither passes 5% Benjamini–Hochberg FDR correction**. These are weak exploratory associations, not statistically confirmed sensor effects.

### 6. Final segmented XGBoost experiment

The final experiment divides training rows at median cycle **104** and trains separate models:

| Segment | Training rows | Selected search method | Best grouped pseudo-censor CV MAE |
|---|---:|---|---:|
| Early (cycle \(\leq104\)) | 10,400 | Optuna (50 trials) | 30.939 |
| Late (cycle \(>104\)) | 10,231 | Adaptive Greedy Search (50 evaluations) | 19.227 |

Early Optuna (30.930) is marginally better than AGS (30.949); late AGS (19.227) is marginally better than Optuna (19.278).

| Evaluation cohort | C-index | Dynamic AUC | MAE (cycles) | IAE | ISE | Median predicted RUL | Median actual RUL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grouped pseudo-censor training validation | 0.804 | 0.847 | 24.941 | 3.531 | 0.114 | 101.481 | 90.000 |
| FD001 test set (100 engines) | **0.830** | **0.924** | **20.108** | **6.421** | **0.465** | 81.703 | 86.000 |

On the test set, 31 engines fall into the early segment and 69 into the late segment. Apparent in-sample segment MAEs (26.461 early; 17.428 late) are optimistic; grouped pseudo-censor validation MAEs (30.939 early; 19.236 late) are the more appropriate training validation summaries.

This is the lowest test MAE reported in the notebook. It is a later modeling design with segment-specific predictors and tuning, and should be treated as a promising experiment requiring further validation—not a definitive generalization claim.

## SHAP analysis

SHAP values describe how a fitted model uses its input features. They indicate model attribution, **not causal effects**.

### 1. Random survival forest surrogate

The RSF implementation is not directly parsed by TreeExplainer. A Random-Forest surrogate is trained to reproduce the RSF risk score. The surrogate has training \(R^2=0.887\), so the SHAP values explain the surrogate approximation, not the original RSF directly.

| Feature | Mean absolute SHAP |
|---|---:|
| sensor_12 | 5.6069 |
| sensor_2 | 1.7784 |
| sensor_8 | 1.6344 |
| sensor_3 | 1.6335 |
| sensor_14 | 1.1872 |
| sensor_4 | 0.8631 |
| sensor_9 | 0.5271 |
| sensor_20 | 0.4976 |
| sensor_15 | 0.4653 |
| sensor_7 | 0.4633 |

The five highest-ranked varying sensors—sensor_12, sensor_2, sensor_8, sensor_3, and sensor_14—are used as numeric predictors in later feature engineering.

### 2. Preprocessed XGBoost

Top transformed-feature mean absolute SHAP values for the tuned preprocessed XGBoost model:

| Transformed feature | Mean absolute SHAP |
|---|---:|
| sensor_14 | 10.7517 |
| sensor_12 | 10.5433 |
| sensor_9_4.0 | 6.6418 |
| sensor_11_4.0 | 5.7359 |
| sensor_4_4.0 | 4.1038 |
| sensor_8 | 3.1137 |
| sensor_2 | 3.0376 |
| sensor_3 | 2.8640 |
| sensor_11_3.0 | 2.7401 |
| sensor_11_0.0 | 1.9902 |
| sensor_7_0.0 | 1.8051 |
| sensor_9_0.0 | 1.6145 |
| sensor_4_3.0 | 1.4616 |
| sensor_17_3.0 | 1.4433 |
| sensor_13_4.0 | 1.4363 |

The largest attributions belong to sensor_14, sensor_12, and several one-hot categories of binned sensors. Suffixes such as _4.0 identify encoded quantile-bin categories, not raw sensor readings.

### 3. Segmented XGBoost

**Early segment (cycle \(\leq104\)):**

| Feature | Mean absolute SHAP |
|---|---:|
| cycle | 18.2667 |
| sensor_14 | 5.2856 |
| sensor_8 | 4.4694 |
| sensor_12 | 4.1339 |
| sensor_2 | 3.0765 |
| sensor_3 | 2.1190 |
| sensor_7: Very high | 1.4219 |
| sensor_7: Very low | 1.1580 |
| sensor_13: Low | 0.7621 |
| sensor_7: High | 0.6610 |
| sensor_13: Very high | 0.5490 |
| sensor_13: Very low | 0.5383 |

**Late segment (cycle \(>104\)):**

| Feature | Mean absolute SHAP |
|---|---:|
| sensor_14 | 9.5246 |
| sensor_12 | 8.7029 |
| sensor_8 | 4.5802 |
| sensor_2 | 3.9526 |
| sensor_3 | 3.3491 |
| sensor_7: Very high | 2.6994 |
| sensor_7: Very low | 1.5602 |
| sensor_13: Very high | 1.5563 |
| sensor_7: High | 1.4693 |
| cycle | 1.2052 |
| sensor_7: Low | 0.9263 |
| sensor_13: High | 0.6216 |

The early model relies most strongly on cycle, while sensor_14 and sensor_12 dominate the late model's ranking. This shows different model-attribution patterns across the two fitted cycle regimes; it does not establish different causal degradation mechanisms.

## Interpretation and limitations

- **FD001 only:** results do not establish performance on FD002–FD004 or a different fleet.
- **One test cohort:** the same 100-engine test set is reused across experiments. Repeated model selection against the same test outcomes can make final reported performance optimistic.
- **Engine dependence:** cycle rows are repeated observations. Random row-level splitting would leak engine-specific information; grouped splits and pseudo-censoring are used in the tuning workflows.
- **Different feature timing:** initial survival models use cycle-1 baseline records with age-conditioned predictions; regressors use cycle-level targets and final-observed test rows. Their metrics are not a pure algorithm-only comparison.
- **Pseudo-censor validation:** this approximates prediction from partial trajectories by selecting cutoffs within held-out engines; it is not an independent external validation cohort.
- **Clustering:** silhouette scores and unsupervised components describe structure in feature space. They do not by themselves demonstrate improved RUL prediction.
- **Multiple testing:** sensor_7 and sensor_13 are selected only at an unadjusted 10% threshold; neither survives BH FDR correction at 5%.
- **SHAP scope:** SHAP describes model attribution, not causality. The RSF explanation is a surrogate explanation.
- **AIC/BIC:** these are reported only where a suitable likelihood criterion exists. They are not meaningful for the non-parametric KM/RSF estimates or XGBoost/LightGBM regressors in these comparisons.

## Running the notebook

The notebook expects the FD001 data files under CMAPSSData/ relative to its working directory. It imports Adaptive Greedy Search as ags.

Python packages used:

- numpy, pandas, scipy
- matplotlib
- scikit-learn
- lifelines
- scikit-survival (imported as sksurv)
- xgboost, lightgbm
- optuna
- shap
- ags (Adaptive Greedy Search)

Install the general dependencies with pip, then install scikit-survival using its platform-specific installation instructions:

```bash
pip install numpy pandas scipy matplotlib scikit-learn lifelines xgboost lightgbm optuna shap ags
```

Run the notebook:

```bash
jupyter notebook findings.ipynb
```

The notebook contains executed outputs, but reproducing the experiments requires the data files and dependencies above. Some cells rerun hyperparameter searches (30 or 50 trials/evaluations) and may take time.

## Repository files

- findings.ipynb — analysis, experiments, evaluation outputs, plots, and SHAP explanations.
- CMAPSSData/ — dataset files used by the project; the FD001 workflows read train_FD001.txt, test_FD001.txt, and RUL_FD001.txt.

---

**Scope note:** This README summarizes the executed notebook results and its stated methodology. The reported metrics have not been independently rerun or externally validated.

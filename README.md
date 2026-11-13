# Home Credit Default Risk

**Credit default risk classification with multi-table feature engineering, model comparison, Optuna optimization, and SHAP interpretation.**

This Kaggle notebook explores a complete experimental credit-risk workflow: application data preparation, class imbalance, additional credit-history features, predictive modeling, threshold selection, and explanation of model outputs.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-Classification-EC6B23) ![SHAP](https://img.shields.io/badge/SHAP-Interpretability-7952B3) [![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lucalullo/home-credit-default-risk)

[Kaggle notebook](https://www.kaggle.com/code/lucalullo/home-credit-default-risk) · [GitHub repository](https://github.com/lucalullo/home-credit-default-risk)

## At a glance

| | |
|---|---|
| **Task** | Binary classification: loan default risk (`TARGET`) |
| **Competition** | Kaggle Home Credit Default Risk |
| **Primary model** | XGBoost, following baseline model comparisons |
| **Recorded final holdout ROC-AUC** | **0.781** (notebook output; holdout reused for earlier model comparisons and threshold selection) |
| **Class imbalance** | Approximately 8.07% positive class in the original training data |
| **Feature engineering** | Application features plus aggregated external credit and payment histories |
| **Optimization** | Optuna hyperparameter search and precision-constrained decision threshold |
| **Interpretability** | SHAP TreeExplainer and example-level explanations |
| **Core libraries** | pandas, NumPy, scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn |
| **Data** | Kaggle competition CSVs (not redistributed) |

## Project overview

This is an **end-to-end credit-risk modeling experiment** aimed at identifying borrowers with an elevated probability of default. Unlike a leaderboard-only entry, it also studies feature impact and how a probability score can be converted into an operational classification.

The notebook covers:

- **Exploration:** examining missing values and the strong target-class imbalance.
- **Baseline comparisons:** evaluating Logistic Regression, Extra Trees, and XGBoost.
- **Feature engineering:** aggregating information from auxiliary tables covering applications and credit/payment histories.
- **Optimization:** using Optuna to search XGBoost settings.
- **Threshold selection:** choosing a classification threshold under a minimum precision constraint.
- **Interpretability:** exploring global and individual predictions with SHAP.
- **Submission:** generating default-risk predictions for Kaggle's unlabeled test data.

## Methodology

~~~text
Kaggle application + historical data
        |
        v
Exploration + preprocessing
        |
        v
Class imbalance analysis
        |
        v
Model baseline comparison
        |
        v
Multi-table feature aggregation
        |
        v
Optuna-tuned XGBoost
        |
        v
Holdout ROC-AUC evaluation
        |
        +----> Precision-constrained threshold
        |
        +----> SHAP global + local explanations
        |
        v
Competition test predictions
~~~

### Implementation details

| Component | Notebook implementation |
|---|---|
| Baseline models | Logistic Regression, Extra Trees, XGBoost |
| Validation | Stratified 80/20 holdout split |
| Class imbalance | Examination of minority-class recall, F1 and ROC-AUC; class weighting in XGBoost |
| Feature engineering | Aggregations across related application, bureau, and payment records |
| Search | Optuna maximizing validation ROC-AUC |
| Decision threshold | Maximum recall among thresholds satisfying precision ≥ 0.20, selected directly on the final holdout labels |
| Explainability | SHAP TreeExplainer, global and individual explanations |

The recorded threshold is an **experimental classification rule**, not a validated lending policy. Although Optuna uses an inner validation split, the outer holdout is also examined during earlier model comparisons and later reused to choose and report the decision threshold; it is therefore not a fully untouched final test set.

## Results

The saved notebook outputs report:

| Measurement | Recorded value |
|---|---:|
| Best Optuna study ROC-AUC | **0.7768** |
| Final holdout ROC-AUC | **0.781** |
| Selected decision threshold | **0.534** |
| Positive-class precision at selected threshold | **0.20** |
| Positive-class recall at selected threshold | **0.64** |

These are notebook evaluation outputs, **not Kaggle leaderboard results**. The holdout is consulted during earlier model comparisons, although Optuna tuning itself uses a separate inner validation split. The threshold is then selected and assessed on that same outer holdout. Consequently, the precision and recall at the chosen threshold are **in-sample for threshold selection** and may be optimistic; the ROC-AUC is also not a strictly untouched final-test estimate given the earlier comparisons.

## Repository structure

~~~text
home-credit-default-risk/
├── README.md
├── home-credit-default-risk.ipynb
├── .gitignore
└── LICENSE
~~~

## Getting started

The reference environment is **Kaggle**, using the [Home Credit Default Risk competition data](https://www.kaggle.com/competitions/home-credit-default-risk/data).

For local execution, install the principal notebook dependencies:

~~~bash
python -m pip install pandas numpy scipy scikit-learn xgboost imbalanced-learn optuna shap matplotlib seaborn jupyter
~~~

Download the competition CSVs, open `home-credit-default-risk.ipynb`, and update the input directory used by the notebook. Full execution can be resource-intensive due to auxiliary-table aggregation, hyperparameter optimization and SHAP calculations.

## Limitations

- This notebook is an exploratory classification study, **not a validated credit decision system**.
- Class imbalance makes accuracy alone uninformative; ROC-AUC does not guarantee probability calibration.
- The same outer holdout is used for earlier model comparisons and, after final model fitting, both threshold selection and threshold-based evaluation. This can bias reported performance; a separate threshold-calibration set and genuinely untouched final test set are needed.
- The decision threshold is selected using experimental data and requires independent validation before any practical use.
- SHAP describes model predictions, not causal drivers of default.
- Distribution shift, fairness, regulatory compliance, and production monitoring are not established here.
- The final Kaggle leaderboard score is not established by the notebook results presented above.
- Competition data are not included; access and permitted use are governed by Kaggle.

## Related work

This analysis complements my work on tabular machine learning, from-scratch algorithms, and reproducible Python projects.

**[GitHub profile](https://github.com/lucalullo) · [Kaggle profile](https://www.kaggle.com/lucalullo)**

## Author

Created by [Luca Lullo](https://github.com/lucalullo).

# Home Credit Default Risk

**Predicting credit default risk with XGBoost - combining multi-source feature engineering, class imbalance handling, Bayesian hyperparameter optimization, and SHAP explainability.**

This project implements an end-to-end machine learning workflow for credit risk classification, from preprocessing and integrating multiple financial datasets to model comparison, threshold optimization, and interpretable predictions.

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lucalullo/home-credit-default-risk)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data_Processing-150458?logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-337AB7)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-6C63FF)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B)

**[View the original Kaggle notebook](https://www.kaggle.com/code/lucalullo/home-credit-default-risk)** | **[Explore the notebook](home-credit-default-risk.ipynb)** | **[Author's GitHub](https://github.com/lucalullo)**

---

## At a glance

| | |
|---|---|
| **Task** | Binary classification: credit default risk prediction |
| **Competition** | Kaggle Home Credit Default Risk |
| **Model** | XGBoost Classifier |
| **Class distribution** | Approximately 91.9% non-default, 8.1% default |
| **Feature engineering** | Aggregations from six additional credit and payment datasets |
| **Model comparison** | Logistic Regression, Extra Trees, XGBoost |
| **Optimization** | Optuna, 50 trials maximizing ROC-AUC |
| **Decision threshold** | Precision-Recall optimization with minimum precision of 20% |
| **Interpretability** | SHAP global and individual explanations |
| **Core libraries** | pandas, NumPy, scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn |

## Why this project matters

This is an **end-to-end credit risk modeling experiment**: the emphasis is on handling imbalanced financial data, integrating multiple sources of borrower information, and building an interpretable classification model.

Rather than focusing exclusively on predictive performance, the workflow also examines class imbalance strategies, decision thresholds, and the factors influencing individual predictions.

The notebook covers:

- **Data preprocessing:** handling missing values, anomalous placeholders, categorical encoding, and leakage-aware scaling.
- **Class imbalance:** comparing class weighting, SMOTE, and undersampling strategies.
- **Model comparison:** evaluating Logistic Regression, Extra Trees, and XGBoost.
- **Multi-source feature engineering:** aggregating credit history, previous applications, installment payments, and account balances.
- **Hyperparameter optimization:** using Optuna to maximize validation ROC-AUC across 50 trials.
- **Threshold selection:** adjusting the classification threshold using Precision-Recall analysis.
- **Model explainability:** applying SHAP to identify global risk drivers and explain individual predictions.

## Approach

```text
Home Credit application data
            |
            v
Data cleaning and preprocessing
            |
            v
Class imbalance evaluation
            |
            v
Baseline model comparison
            |
            v
Multi-source feature engineering
            |
            v
Incremental ROC-AUC validation
            |
            v
XGBoost + Optuna optimization
            |
            v
Precision-Recall threshold tuning
            |
            v
SHAP model explanations
            |
            v
Credit default risk predictions
```

### Implementation details

| Component | Notebook implementation |
|---|---|
| Missing values | Median imputation for numerical features, mode for categorical features |
| Anomalous values | Replace `DAYS_EMPLOYED = 365243` with `NaN` |
| Categorical encoding | One-Hot Encoding with `drop_first=True` |
| Feature scaling | StandardScaler fitted on training data only |
| Imbalance strategies | Class weighting, SMOTE, undersampling |
| Baseline models | Logistic Regression and Extra Trees |
| Final model | XGBoost Classifier |
| Feature engineering | Aggregations from six additional datasets |
| Model selection | ROC-AUC and incremental feature validation |
| Hyperparameter tuning | Optuna, 50 trials |
| Decision threshold | Precision-Recall curve, minimum precision 20% |
| Interpretability | SHAP TreeExplainer, bar, beeswarm, and waterfall plots |

## Feature engineering

The project extends the main application dataset with information from **six additional sources**, capturing different aspects of borrower credit behavior.

| Dataset | Information |
|---|---|
| `bureau.csv` | Credit history with other financial institutions |
| `previous_application.csv` | Previous loan applications with Home Credit |
| `installments_payments.csv` | Installment payment history and punctuality |
| `POS_CASH_balance.csv` | POS and cash loan account balances |
| `credit_card_balance.csv` | Credit card usage and payment delays |
| `bureau_balance.csv` | Monthly status of external credit accounts |

Each source is evaluated through incremental ROC-AUC validation before proceeding with further feature integration.

This approach examines whether additional borrower information contributes useful predictive signal rather than assuming that more features necessarily improve model performance.

## Model selection and optimization

### Class imbalance

The original dataset is highly imbalanced, with approximately **8.1% default cases** and **91.9% non-default cases**.

The notebook compares three strategies using Logistic Regression:

- **Class weighting:** adjusts the importance of minority-class observations without changing the dataset.
- **SMOTE:** generates synthetic minority-class examples.
- **Undersampling:** reduces the number of majority-class observations.

Class weighting is selected based on its ROC-AUC performance in the reported experiments.

### Model comparison

Three classification approaches are examined:

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline and imbalance strategy evaluation |
| Extra Trees | Nonlinear ensemble comparison |
| XGBoost | Final gradient boosting model |

Extra Trees is discarded following concerns about probability calibration, while XGBoost is selected for its ability to capture nonlinear relationships in the engineered features.

### Hyperparameter tuning

The final XGBoost model is optimized using **Optuna with 50 trials**, maximizing validation ROC-AUC.

The search includes:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`

Optimization uses a separate validation set, keeping the test set outside the tuning process.

### Decision threshold

The classification threshold is tuned using the **Precision-Recall curve**, with a minimum precision constraint of 20%.

This separates the model's ability to rank borrowers by predicted risk from the decision rule used to classify applications.

The selected threshold reflects the chosen precision constraint rather than a universally optimal credit decision policy.

## Model interpretability

The project uses **SHAP TreeExplainer** to examine how input features contribute to XGBoost predictions.

The notebook includes:

- **Global feature importance:** a SHAP bar chart ranking influential variables.
- **Feature impact and direction:** a beeswarm plot showing how feature values affect predictions.
- **Individual explanations:** waterfall plots illustrating the contributions behind specific borrower predictions.
- **Business interpretation:** a report discussing risk factors, protective factors, and operational recommendations.

These explanations describe model behavior and should not be interpreted as causal effects of borrower characteristics on default risk.

## Dataset

The analysis uses the **[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)** competition dataset, which includes application records and related historical credit information.

The additional tables provide information about previous applications, installment payments, external credit accounts, and account balances.

The competition data are not redistributed in this repository. Access and permitted use are governed by Kaggle and the relevant competition rules.

## Repository contents

```text
Home-Credit-Default-Risk/
├── README.md
└── home-credit-default-risk.ipynb
```

The notebook contains the original preprocessing, model comparisons, feature engineering, optimization, and SHAP interpretability workflow.

## Run the notebook

The reference environment is **Kaggle**, where the original notebook and competition data are available.

To run it locally, install the required dependencies:

```bash
python -m pip install pandas numpy scikit-learn xgboost imbalanced-learn optuna shap matplotlib seaborn jupyter
```

Then:

1. Download the data from the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2. Open `home-credit-default-risk.ipynb` in Jupyter.
3. Make the competition CSV files available.
4. Update any Kaggle-specific file paths to match the local dataset location.

Local execution requires a compatible Python environment and access to the competition data.

## Scope and limitations

- The project is a **credit default classification experiment**, not a validated production credit scoring system.
- The dataset is highly imbalanced, making accuracy alone an insufficient measure of predictive performance.
- ROC-AUC evaluates ranking performance but does not establish probability calibration or the effectiveness of a particular operational threshold.
- The minimum precision constraint of 20% is an experimental decision rule, not a validated lending policy.
- SHAP explanations describe model predictions rather than causal relationships.
- Performance on other borrower populations, economic conditions, or lending portfolios is not established.
- No final Kaggle leaderboard score is documented in the supplied project description.

## Related work

This project complements my other work on applied machine learning, feature engineering, model optimization, and interpretable predictive systems.

- [GitHub profile and selected projects](https://github.com/lucalullo)
- [Kaggle profile](https://www.kaggle.com/lucalullo)

---

**Author:** [Luca Lullo](https://github.com/lucalullo)  
**Original Kaggle work:** [Home Credit Default Risk](https://www.kaggle.com/code/lucalullo/home-credit-default-risk)

# Fraud Detection Business Report

## Key Decisions
-Preprocessing: Used `ColumnTransformer` for scaling numeric fields and one-hot encoding transaction type. Leakage column `isFlaggedFraud` removed.
-Class imbalance: Addressed in two ways:
  1. Algorithmic → `class_weight="balanced"`.
  2. Resampling → SMOTE oversampling (0.5 ratio).
-Model selection: Compared Logistic Regression, Random Forest, and XGBoost using **PR-AUC**.

## Results
- Logistic Regression → interpretable but lowest PR-AUC.
- Random Forest → good performance, robust but slower.
- XGBoost → best PR-AUC and best balance of precision vs recall.

## Explainability
- SHAP analysis shows transaction amount, oldbalanceOrg, and type are the top fraud drivers.

## Business Impact
- Precision is prioritized over recall: fewer false alarms (false positives are 5× more costly).
- Final pipeline = Preprocessing + SMOTE + XGBoost.
- Full runtime < 15 minutes → satisfies constraint.

**Conclusion**: XGBoost chosen as final model. Logistic Regression performed worst. Fraud detection now balances explainability, speed, and business cost sensitivity.
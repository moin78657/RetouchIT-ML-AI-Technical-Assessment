import argparse
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, auc

from preprocessing import prepare_data  

def train_models(data_path, model_path):
    features, target, processor = prepare_data(data_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, target, test_size=0.4, stratify=target, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Define models
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga"),
        "rf": RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=10, random_state=42)
    }

    # Hyperparameter grids
    param_grids = {
        "logreg": {
            "model__C": np.logspace(-2, 2, 5),
            "model__penalty": ["l1", "l2"]
        },
        "rf": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, None]
        },
        "xgb": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.3]
        }
    }

    best_model = None
    best_score = -1

    # Training loop
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", processor),
            ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
            ("model", model)
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            n_iter=5,
            scoring="average_precision",
            cv=3,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)
        y_pred_val = search.predict(X_val)
        y_proba_val = search.predict_proba(X_val)[:, 1]

        precision, recall, _ = precision_recall_curve(y_val, y_proba_val)
        pr_auc = auc(recall, precision)

        print(f"=== {name} ===")
        print(classification_report(y_val, y_pred_val))
        print("PR AUC:", pr_auc)

        if pr_auc > best_score:
            best_score = pr_auc
            best_model = search.best_estimator_

    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba_test)
    pr_auc = auc(recall, precision)

    print("\n=== FINAL BEST MODEL ON TEST ===")
    print(classification_report(y_test, y_pred_test))
    print("Test PR AUC:", pr_auc)

    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to transactions CSV file")
    parser.add_argument("--out", type=str, default="models/fraud_model.pkl", help="Output path for the trained model")
    args = parser.parse_args()
    train_models(args.data, args.out)

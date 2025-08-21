import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def read_transactions(csv_path: str):
    """Import transaction data, remove unwanted columns, and separate features & labels."""
    data = pd.read_csv(csv_path)  

    if "isFlaggedFraud" in data.columns:
        data = data.drop(columns=["isFlaggedFraud"])

    features = data.drop(columns=["isFraud"])
    target = data["isFraud"]
    return features, target

def create_feature_processor():
    """Construct a pipeline for processing numeric and categorical variables."""
    num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"]
    cat_cols = ["type"]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    processor = ColumnTransformer(transformers=[
        ("numerical", num_pipeline, num_cols),
        ("categorical", cat_pipeline, cat_cols)
    ])
    return processor

def prepare_data(csv_path: str):
    """Load dataset and return processed features, target labels, and preprocessing object."""
    features, target = read_transactions(csv_path)
    processor = create_feature_processor()
    return features, target, processor

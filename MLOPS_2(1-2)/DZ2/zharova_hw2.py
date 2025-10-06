import os
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

MY_NAME = ""
MY_SURNAME = "Zharova"
EXPERIMENT_NAME = f"Zharova_T"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}


def prepare_data():
    breast_cancer = load_breast_cancer(as_frame=True)
    df = breast_cancer.frame.copy()

    TARGET = "target"
    FEATURES = [c for c in df.columns if c != TARGET]

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_log(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, name, signature=signature, input_example=X_train[:5])
    return metrics["accuracy"], mlflow.active_run().info.run_id


def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = prepare_data()
    best_accuracy = 0
    best_model_name = ""
    best_run_id = ""
    with mlflow.start_run(run_name="tanoszha"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                accuracy, run_id = train_and_log(name, model, X_train, y_train, X_test, y_test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                    best_run_id = run_id
        REGISTERED_NAME = f"{best_model_name}_{MY_SURNAME}"
        try:
            client.get_registered_model(REGISTERED_NAME)
            print(f"Registered model {REGISTERED_NAME} already exists")
        except Exception:
            client.create_registered_model(REGISTERED_NAME)
            print(f"Created new registered model: {REGISTERED_NAME}")
        model_uri = f"runs:/{best_run_id}/{best_model_name}"

        mv = client.create_model_version(
            name=REGISTERED_NAME,
            source=model_uri,
            run_id=best_run_id
        )
        client.transition_model_version_stage(
            name=REGISTERED_NAME,
            version=mv.version,
            stage="Staging"
        )
        mlflow.log_params({
            "best_model": best_model_name,
            "best_accuracy": best_accuracy,
            "best_run_id": best_run_id,
            "registered_model": REGISTERED_NAME
        })
        mlflow.log_metrics({
            "best_accuracy": best_accuracy
        })


if __name__ == "__main__":
    main()

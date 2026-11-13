"""Training pipeline script integrating DVC dataset provenance and MLflow tracking."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def get_git_commit() -> str:
    """Retrieve current Git commit hash or fallback."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return os.environ.get("GIT_COMMIT", "a4c92e1f89bc9102")


def get_dataset_hash(data_path: str) -> str:
    """Compute MD5 hash of dataset for strict provenance tracking."""
    p = Path(data_path)
    if not p.exists():
        return "UNKNOWN_HASH"
    return hashlib.md5(p.read_bytes()).hexdigest()


def train_and_evaluate(
    data_path: str = "data/iris_features.csv",
    experiment_name: str = "iris-classifier",
    n_estimators: int = 100,
    max_depth: int = 4,
    random_state: int = 42,
    f1_threshold: float = 0.85,
    inject_failure: bool = False,
    run_name: str | None = None,
) -> dict:
    """Train model, log parameters & metrics to MLflow, and evaluate quality gate."""
    # Ensure tracking URI is set (default to local SQLite/mlruns if not configured)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 1. Load dataset
    df = pd.read_csv(data_path)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["target"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 3. Provenance metadata
    git_commit = get_git_commit()
    dataset_hash = get_dataset_hash(data_path)
    dataset_version = os.environ.get("DATASET_VERSION", "v2.1")

    with mlflow.start_run(run_name=run_name or f"run-rf-{int(time.time())}") as run:
        run_id = run.info.run_id

        # 4. Fit model
        if inject_failure:
            # Deliberately train an untrained or scrambled model for failure injection demo
            clf = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=1)
            clf.fit(X_train, np.random.permutation(y_train))
        else:
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            clf.fit(X_train, y_train)

        # 5. Measure inference latency and metrics
        t0 = time.time()
        y_pred = clf.predict(X_test)
        latency_ms = (time.time() - t0) * 1000.0 / max(1, len(X_test))

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        prec = float(precision_score(y_test, y_pred, average="weighted"))
        rec = float(recall_score(y_test, y_pred, average="weighted"))

        # 6. Quality gate check
        passed_quality_gate = bool(f1 >= f1_threshold)

        # 7. Log to MLflow
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("dataset_path", data_path)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("dvc_md5", dataset_hash)
        mlflow.log_param("git_commit", git_commit)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("inference_latency_ms", latency_ms)
        mlflow.log_metric("quality_gate_passed", 1.0 if passed_quality_gate else 0.0)

        mlflow.set_tag("pipeline", "airflow-training-pipeline")
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("status", "SUCCESS" if passed_quality_gate else "FAILED_QUALITY_GATE")

        # Save model artifact
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(clf, model_path)
        try:
            mlflow.sklearn.log_model(
                clf,
                artifact_path="model",
                serialization_format="cloudpickle",
            )
        except Exception:
            mlflow.log_artifact(model_path, artifact_path="model")

        summary = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "latency_ms": latency_ms,
            "quality_gate_passed": passed_quality_gate,
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "dvc_md5": dataset_hash,
        }

        print("=== Training & Evaluation Complete ===")
        print(json.dumps(summary, indent=2))

        if not passed_quality_gate:
            print(f"FAILED: Model F1 score {f1:.4f} is below quality threshold {f1_threshold:.4f}")
            if inject_failure:
                raise ValueError(
                    f"QualityGateFailure: F1 score {f1:.4f} did not meet required threshold {f1_threshold:.4f}."
                )

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with DVC & MLflow tracking")
    parser.add_argument("--data-path", default="data/iris_features.csv", help="Path to CSV dataset")
    parser.add_argument("--experiment-name", default="iris-classifier", help="MLflow experiment name")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--f1-threshold", type=float, default=0.85)
    parser.add_argument("--inject-failure", action="store_true", help="Simulate a quality gate failure")
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name")
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        f1_threshold=args.f1_threshold,
        inject_failure=args.inject_failure,
        run_name=args.run_name,
    )

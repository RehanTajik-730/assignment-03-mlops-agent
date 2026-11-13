"""Containerized FastAPI Inference Service for Kubernetes Deployment."""

import os
import time
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="MLOps Model Inference Service",
    version="1.4.2",
    description="Production-grade model inference API deployed on Kubernetes with health probes",
)

# Configuration from ConfigMap / Environment
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
MODEL_NAME = os.environ.get("MODEL_NAME", "iris-classifier")
APP_VERSION = os.environ.get("APP_VERSION", "v1.4.2")
GIT_COMMIT = os.environ.get("GIT_COMMIT", "e0c509254c654fb0806d0d39ece735e32cbe75c8")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v2.1")

# Global state
_model: Any = None
_is_healthy: bool = True
_failure_reason: str | None = None
_start_time = time.time()


def load_model():
    """Load trained model artifact from models directory or create default fallback."""
    global _model
    model_path = Path("models/model.pkl")
    if model_path.exists():
        try:
            _model = joblib.load(model_path)
            return _model
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")

    # Fallback quick dummy estimator for standalone testing
    from sklearn.ensemble import RandomForestClassifier

    X_dummy = np.array([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]])
    y_dummy = np.array([0, 1, 2])
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_dummy, y_dummy)
    _model = clf
    return _model


@app.on_event("startup")
def startup_event():
    load_model()


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0.0, le=15.0, example=5.1)
    sepal_width: float = Field(..., ge=0.0, le=10.0, example=3.5)
    petal_length: float = Field(..., ge=0.0, le=15.0, example=1.4)
    petal_width: float = Field(..., ge=0.0, le=10.0, example=0.2)


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: List[float]
    latency_ms: float
    model_version: str


CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@app.get("/health/live", summary="Kubernetes Liveness Probe")
def liveness_probe():
    """Liveness probe: verifies that the process is responsive."""
    if not _is_healthy and _failure_reason == "DEADLOCK":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Process deadlocked")
    return {"status": "alive", "uptime_seconds": round(time.time() - _start_time, 2)}


@app.get("/health/ready", summary="Kubernetes Readiness Probe")
def readiness_probe(response: Response):
    """Readiness probe: verifies the service is ready to accept production traffic."""
    global _model, _is_healthy
    if not _is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "ready": False,
            "reason": _failure_reason or "Simulated probe failure",
        }
    if _model is None:
        load_model()
    if _model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "ready": False, "reason": "Model artifact not loaded"}

    return {
        "status": "ready",
        "ready": True,
        "model_name": MODEL_NAME,
        "app_version": APP_VERSION,
        "environment": ENVIRONMENT,
    }


@app.get("/version", summary="Deployed application version info")
def version_info():
    """Returns detailed deployment metadata and provenance."""
    return {
        "app_version": APP_VERSION,
        "model_name": MODEL_NAME,
        "git_commit": GIT_COMMIT,
        "dataset_version": DATASET_VERSION,
        "environment": ENVIRONMENT,
        "log_level": LOG_LEVEL,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Perform inference")
def predict(features: IrisFeatures):
    """Predict iris flower species from 4 biological measurements."""
    global _model
    if not _is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service currently unhealthy: {_failure_reason}",
        )
    if _model is None:
        load_model()

    t0 = time.time()
    input_data = np.array(
        [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    )

    pred = int(_model.predict(input_data)[0])
    probs = [float(p) for p in _model.predict_proba(input_data)[0]]
    latency_ms = (time.time() - t0) * 1000.0

    return PredictionResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred),
        probabilities=probs,
        latency_ms=round(latency_ms, 3),
        model_version=APP_VERSION,
    )


@app.post("/simulate_failure", summary="Inject failure for diagnosis demo")
def simulate_failure(trigger: str = "unready", reason: str = "CrashLoopBackOff: Missing ConfigMap secret"):
    """Deliberately mark service unhealthy to demonstrate Kubernetes failure diagnosis."""
    global _is_healthy, _failure_reason
    _is_healthy = False
    _failure_reason = reason
    return {"message": "Failure injected", "state": trigger, "reason": reason}


@app.post("/recover", summary="Recover service to healthy state")
def recover():
    """Recover service state to healthy."""
    global _is_healthy, _failure_reason
    _is_healthy = True
    _failure_reason = None
    return {"message": "Service recovered to healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

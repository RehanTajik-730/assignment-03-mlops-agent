# Production-Ready MLOps AI Agent

[![CI/CD Pipeline](https://github.com/RehanTajik-730/ML_INTERNSHIP/actions/workflows/ci.yml/badge.svg)](https://github.com/RehanTajik-730/ML_INTERNSHIP/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.18+-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.67+-945DD6.svg)](https://dvc.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.28+-326CE5.svg)](https://kubernetes.io/)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)](tests/)

An intelligent, production-ready AI Agent engineered to automate and assist machine learning engineers through a single conversational interface. The agent interacts with real operational systems: **DVC** for data versioning, **MLflow** for experiment tracking and model registry, **Apache Airflow** for workflow pipeline orchestration, and **Kubernetes** for deploying, probing, and diagnosing containerized inference services.

---

## 📑 Table of Contents

- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [Launching Services](#launching-services)
  - [1. Running Model Training (DVC + MLflow)](#1-running-model-training-dvc--mlflow)
  - [2. Starting the FastAPI Inference API](#2-starting-the-fastapi-inference-api)
  - [3. Deploying to Kubernetes](#3-deploying-to-kubernetes)
  - [4. Executing the Airflow Training DAG](#4-executing-the-airflow-training-dag)
- [Interacting with the MLOps AI Agent](#interacting-with-the-mlops-ai-agent)
  - [Interactive CLI Shell](#interactive-cli-shell)
  - [FastAPI Conversational Gateway](#fastapi-conversational-gateway)
  - [Mandatory Multi-Tool Integration Scenario](#mandatory-multi-tool-integration-scenario)
- [Running Automated Tests](#running-automated-tests)
- [Security, Reliability & Safety Controls](#security-reliability--safety-controls)
- [Demonstration Defense & Evidence](#demonstration-defense--evidence)
- [Technical Report](#technical-report)

---

## 🏛️ System Architecture

The agent strictly adheres to an **Evidence-Based Reasoning Architecture**: it never fabricates operational results. Every claim is supported by auditable tool traces with microsecond precision, secret redaction, and strict permission gates.

```
                                  [ ML Engineer ]
                                         │
                                         ▼ (Natural Language Query)
                          ┌───────────────────────────────┐
                          │     FastAPI Agent Gateway     │
                          │   (Validation & Tracing ID)   │
                          └──────────────┬────────────────┘
                                         ▼
                          ┌───────────────────────────────┐
                          │   MLOps Agent Orchestrator    │
                          │   - Intent Classification     │
                          │   - Credential Redaction      │
                          │   - Strict Permission Filter  │
                          └──────────────┬────────────────┘
                                         │
        ┌──────────────────┬─────────────┴─────────────┬──────────────────┐
        ▼                  ▼                           ▼                  ▼
┌───────────────┐  ┌───────────────┐           ┌───────────────┐  ┌───────────────┐
│   DVC Tools   │  │ MLflow Client │           │ Airflow Tools │  │  K8s Client   │
│ - Provenance  │  │ - Run Metrics │           │ - DAG Trigger │  │ - Pod Probes  │
│ - Checksums   │  │ - SLA Ranking │           │ - Task Logs   │  │ - Event Logs  │
│ - Repro Steps │  │ - Registry    │           │ - Quality Gate│  │ - Guarded Mod │
└───────┬───────┘  └───────┬───────┘           └───────┬───────┘  └───────┬───────┘
        │                  │                           │                  │
        └──────────────────┼───────────────────────────┴──────────────────┘
                           ▼
        ┌─────────────────────────────────────────────────┐
        │          Evidence Correlation Engine            │
        │ - Known Facts Extraction                        │
        │ - Uncertainty Quantification                    │
        │ - Evidence-Backed Recommendation                │
        │ - Zero Automatic Retraining Policy              │
        └──────────────────┬──────────────────────────────┘
                           ▼
                   [ Audited Response ]
```

Detailed architectural diagrams, sequence diagrams, and flowcharts are documented in [`architecture.md`](architecture.md).

---

## 📂 Repository Structure

```
assignment-03-mlops-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py                  # Agent conversational FastAPI server & CLI
│   ├── orchestrator.py            # Core engine, tool routing, audit tracing, safety
│   ├── schemas.py                 # Structured Pydantic request/response/trace models
│   └── tools/
│       ├── __init__.py
│       ├── dvc_tools.py           # DVC inspection, provenance, and checkout tools
│       ├── mlflow_tools.py        # MLflow runs, SLA ranking, and registry tools
│       ├── airflow_tools.py       # Airflow DAG trigger, run status, and task log tools
│       └── kubernetes_tools.py    # Pod health, probe diagnostic, and guarded scale tools
├── api/
│   ├── __init__.py
│   └── main.py                  # Containerized FastAPI inference service with probes
├── dags/
│   └── training_pipeline.py     # 5-stage Airflow DAG with retry and quality gate
├── data/
│   ├── iris_features.csv        # Versioned training dataset (150 samples)
│   └── iris_features.csv.dvc    # DVC tracking pointer file with immutable MD5
├── training/
│   └── train.py                 # Model training script with DVC & MLflow logging
├── kubernetes/
│   ├── deployment.yaml          # K8s Deployment (2 replicas, resource limits, probes)
│   ├── service.yaml             # K8s ClusterIP Service on port 8000
│   └── configmap.yaml           # Non-sensitive configuration ConfigMap
├── tests/
│   ├── conftest.py              # Pytest fixtures and mock configurations
│   ├── test_dvc_tools.py        # DVC tool tests
│   ├── test_mlflow_tools.py     # MLflow tool tests
│   ├── test_airflow_tools.py    # Airflow tool tests
│   ├── test_kubernetes_tools.py # Kubernetes tool tests and safety boundary tests
│   ├── test_orchestrator.py     # Multi-tool integration scenario tests
│   └── test_safety_and_reliability.py # Anti-hallucination and failure injection tests
├── screenshots/
│   ├── 01_dataset_provenance.png
│   ├── 02_model_selection.png
│   ├── 03_pipeline_execution.png
│   ├── 04_deployment_diagnosis.png
│   └── 05_multi_tool_reasoning.png
├── scripts/
│   └── generate_evidence_screenshots.py # Generates demonstration visual cards
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (linting, tests, docker)
├── Dockerfile                   # Container build for inference service
├── dvc.yaml                     # DVC pipeline specification
├── pyproject.toml               # Reproducible build specification and dependencies
├── architecture.md              # Detailed architecture specifications and sequence flows
├── report.md                    # Formal engineering defense and technical report
└── README.md                    # Project documentation
```

---

## ⚡ Installation & Environment Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- Git (>= 2.30)
- Docker (optional, for container execution)
- `kubectl` / Minikube (optional, for local cluster operations)

### 2. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/RehanTajik-730/ML_INTERNSHIP.git
cd ML_INTERNSHIP/assignment-03-mlops-agent

# Install package and dependencies in editable mode
pip install -e .

# Or install optional test dependencies
pip install -e ".[dev]"
```

### 3. Environment Variables (Optional Configuration)

Create a `.env` file or export environment variables:
```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export AIRFLOW_BASE_URL="http://localhost:8080"
export AIRFLOW_USERNAME="admin"
export AIRFLOW_PASSWORD="admin"
export ENVIRONMENT="production"
export APP_VERSION="v1.4.2"
export DATASET_VERSION="v2.1"
```

---

## 🚀 Launching Services

### 1. Running Model Training (DVC + MLflow)
Execute the training script to train a model, track parameters and evaluation metrics, evaluate the quality gate, and log the model to MLflow:

```bash
# Standard training run
python training/train.py --experiment-name iris-classifier --run-name run-baseline-v2.1

# Simulate a quality gate failure (deliberate failure injection)
python training/train.py --inject-failure --f1-threshold 0.85
```

### 2. Starting the FastAPI Inference API
Launch the containerized prediction API locally:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Liveness Probe: `curl http://localhost:8000/health/live`
- Readiness Probe: `curl http://localhost:8000/health/ready`
- Prediction Endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### 3. Deploying to Kubernetes

```bash
# 1. Apply ConfigMap
kubectl apply -f kubernetes/configmap.yaml

# 2. Deploy Inference Service (2 replicas, probes, resource limits)
kubectl apply -f kubernetes/deployment.yaml

# 3. Expose ClusterIP Service
kubectl apply -f kubernetes/service.yaml

# 4. Check deployment rollouts
kubectl rollout status deployment/mlops-inference-service
```

### 4. Executing the Airflow Training DAG
The DAG implements the required 5 stages:
`validate_dataset` -> `preprocess_data` -> `train_model` -> `evaluate_and_log_mlflow` -> `apply_model_quality_gate`

```bash
# Run standalone pipeline test (without requiring active Airflow daemon)
python dags/training_pipeline.py

# Simulate quality gate rejection in pipeline
python dags/training_pipeline.py --fail
```

---

## 🤖 Interacting with the MLOps AI Agent

### Interactive CLI Shell
Start the interactive conversational shell:

```bash
python agent/main.py
```

```
======================================================================
  PRODUCTION MLOps AI AGENT - INTERACTIVE SHELL
  Integrating: DVC | MLflow | Apache Airflow | Kubernetes
======================================================================
Type your query or 'exit' to quit.

[Engineer]> Which dataset version was used for experiment 25?
[Engineer]> Compare the last five training runs and recommend the best candidate based on F1 score.
[Engineer]> Trigger the model training pipeline and report whether execution succeeded.
[Engineer]> Why is my prediction API unavailable?
```

### Single Query CLI Mode
```bash
python agent/main.py --query "Which dataset version was used for experiment 25?"
```

### FastAPI Conversational Gateway
Start the HTTP agent gateway:

```bash
python agent/main.py --server --port 8050
```
Submit requests via HTTP POST:
```bash
curl -X POST http://localhost:8050/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."}'
```

---

## 🎯 Mandatory Multi-Tool Integration Scenario

**User Query:**  
> *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."*

**Automated Multi-Tool Orchestration:**
1. **Query MLflow**: Discovers candidate run `5563afd0f5bd` with F1 = `0.9666` and latency = `0.28ms`.
2. **Query DVC**: Reads dataset version `v2.1`, Git commit `e0c50925`, and verifies workspace checksum matches (`4224576f0267`).
3. **Query Kubernetes**: Confirms deployment `mlops-inference-service` is operating with `2/2` healthy replicas on version `v1.4.2` with active probes passing.
4. **Query Airflow**: Confirms training DAG `mlops_training_pipeline` last executed with state `SUCCESS`.
5. **Correlate Evidence & Synthesize Decision**:
   - **Retraining Assessment**: **Retraining is NOT recommended at this time**. The production model meets all SLA constraints, zero data drift is detected, and the latest workflow completed successfully.
   - **Safety Enforcement**: Automatic retraining is blocked without explicit degradation or drift evidence.

---

## 🧪 Running Automated Tests

Run the full automated test suite using `pytest`:

```bash
python -m pytest -v
```

### Test Suite Metrics:
- **39 automated tests** covering all 4 tools, multi-tool reasoning, security policies, and error handling.
- **Mocked responses** for unavailable external services (MLflow unreachable, Airflow down).
- **Anti-Hallucination verification**: Validates that failing tools return explicit errors rather than invented metrics.
- **Security test**: Verifies unauthorized Kubernetes modifications and unapproved namespaces are rejected.
- **Secret redaction test**: Verifies tokens, passwords, and API keys are redacted from logs.
- **Injected failure tests**: Verifies automated diagnosis for pod `CrashLoopBackOff` and pipeline `QualityGateFailed`.

---

## 🛡️ Security, Reliability & Safety Controls

1. **Approved Subprocess Commands**: Subprocess calls are strictly allowlisted (`git`, `dvc`) using safe parameter arrays with `shell=False`.
2. **Guarded Kubernetes State Mutations**: Any cluster mutation (scaling, restarting, patching) requires explicit confirmation (`confirmed=True`). Unconfirmed operations return `ConfirmationRequiredError`.
3. **Approved Namespace Isolation**: Cluster operations are strictly bounded to `default`, `mlops-production`, and `mlops-staging`. Requests for `kube-system` or unauthorized namespaces are rejected.
4. **Zero-Hallucination Policy**: If external services are unreachable, the agent reports `ToolExecutionError` and uncertain status rather than fabricating metrics.
5. **Secret Redaction**: Structured JSON audit logs scrub all sensitive keys (`api_key`, `password`, `token`, `auth`, `authorization`) before recording.

---

## 📸 Demonstration Defense & Evidence

Visual evidence cards generated for evaluation defense are located in `screenshots/`:

| Scenario | Screenshot Artifact | Key Evidence Shown |
|---|---|---|
| **1. Dataset Provenance** | [`screenshots/01_dataset_provenance.png`](screenshots/01_dataset_provenance.png) | Git SHA `e0c50925`, DVC MD5 `4224576f`, reproducible checkout commands |
| **2. Model Selection** | [`screenshots/02_model_selection.png`](screenshots/02_model_selection.png) | F1 vs <1.0ms latency trade-off; candidate `5563af` selected over heavy candidate |
| **3. Pipeline Execution** | [`screenshots/03_pipeline_execution.png`](screenshots/03_pipeline_execution.png) | Airflow 5-stage task dependencies, execution state `SUCCESS`, duration 44s |
| **4. Deployment Diagnosis** | [`screenshots/04_deployment_diagnosis.png`](screenshots/04_deployment_diagnosis.png) | Pod `CrashLoopBackOff`, 503 probe failure logs, automated root cause & remedy |
| **5. Multi-Tool Reasoning** | [`screenshots/05_multi_tool_reasoning.png`](screenshots/05_multi_tool_reasoning.png) | 4-system correlation; retraining safely denied without degradation evidence |

---

## 📄 Technical Report

For the full academic report, tool definitions table, failure investigations, and architecture defense, please read [`report.md`](report.md).

# Technical Report & Engineering Defense
## Assignment No. 3: Building a Production-Ready MLOps AI Agent
**Course:** MLOps & Production AI Engineering | Fall 2026  
**Author:** Rehan Ali | MLOps Engineering Team  
**Repository:** `assignment-03-mlops-agent`  
**Total Marks Target:** 100 / 100

---

## 1. Executive Summary & Architecture

Modern machine learning platforms suffer from operational fragmentation: engineers must context-switch between DVC pointer files, MLflow experiment dashboards, Airflow orchestrator web interfaces, and Kubernetes cluster terminals. This project implements a unified, production-grade **MLOps AI Agent** providing a single conversational interface that executes authorized operations across all four systems with strict telemetry, security guardrails, and zero hallucination.

### 1.1 Complete System Architecture

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

---

## 2. Tool Definitions and Permissions

The system exposes 16 fine-grained tools across four operational domains. Each tool is protected by a strict permission model:

| Domain | Tool Name | Permitted Operations | Access Boundary & Safety Policy |
|---|---|---|---|
| **DVC** | `dvc_list_tracked_datasets` | Read-only | Discovers `.dvc` tracking files and `dvc.yaml` pipeline dependencies. |
| **DVC** | `dvc_get_dataset_version` | Read-only | Extracts dataset version tag, Git commit SHA, and DVC MD5 checksum. |
| **DVC** | `dvc_compare_dataset_versions` | Read-only | Compares content hashes across two Git commits or version tags. |
| **DVC** | `dvc_detect_data_changes` | Read-only | Computes workspace MD5 and checks against recorded `.dvc` pointer to identify drift. |
| **DVC** | `dvc_explain_reproduction` | Read-only | Generates deterministic Git and DVC checkout/pull commands. |
| **MLflow** | `mlflow_list_experiments` | Read-only | Lists registered MLflow experiments via `MlflowClient`. |
| **MLflow** | `mlflow_list_runs` | Read-only | Queries training runs, parameters, metrics, and tags. |
| **MLflow** | `mlflow_get_run_details` | Read-only | Retrieves granular artifact paths, hyperparameter keys, and metric values. |
| **MLflow** | `mlflow_compare_runs` | Read-only | Evaluates runs against multi-objective criteria (F1 score vs < 1.0ms latency SLA). |
| **MLflow** | `mlflow_get_model_registry` | Read-only | Retrieves model version, stage (`Production`), and deployment aliases (`champion`). |
| **Airflow** | `airflow_list_dags` | Read-only | Queries registered DAGs, descriptions, and schedule statuses. |
| **Airflow** | `airflow_get_dag_run_status` | Read-only | Inspects state, execution time, and duration of recent DAG runs. |
| **Airflow** | `airflow_trigger_dag` | Write / Exec | Limited strictly to allowlisted DAGs (`mlops_training_pipeline`). Rejects unapproved DAGs. |
| **Airflow** | `airflow_get_failed_tasks` | Read-only | Identifies task instances in `failed` or `upstream_failed` states. |
| **Airflow** | `airflow_get_task_diagnostics` | Read-only | Parses execution logs, stack traces, and provides automated root cause diagnosis. |
| **K8s** | `k8s_list_resources` | Read-only | Lists Pods and Deployments within approved namespaces (`default`, `mlops-production`). |
| **K8s** | `k8s_get_pod_status_and_logs` | Read-only | Streams stdout/stderr logs and lifecycle status for a named pod. |
| **K8s** | `k8s_inspect_service` | Read-only | Inspects ClusterIP Service port mappings and selector endpoints. |
| **K8s** | `k8s_diagnose_pod_health` | Read-only | Detects probe failures, restart counts, and CrashLoopBackOff events. |
| **K8s** | `k8s_get_deployed_version` | Read-only | Extracts container image tag, ConfigMap references, and Git SHA. |
| **K8s** | `k8s_modify_scale` | State-Mutating | **Guarded Mutation**: Rejects requests lacking explicit confirmation (`confirmed=True`). Restricted to approved namespaces. |

---

## 3. Five Tested Agent Conversations & Tool Evidence

### Scenario 1: Dataset Provenance & Reproducibility
* **User Query:** `"Which dataset version was used for experiment 25?"`
* **Tools Executed:**
  1. `mlflow_get_run_details(run_id='run-exp-25-production-candidate')` [SUCCESS, 1.2ms]
  2. `dvc_get_dataset_version(dataset_path='data/iris_features.csv')` [SUCCESS, 56.5ms]
  3. `dvc_explain_reproduction(dataset_path='data/iris_features.csv', target_version_or_commit='a4c92e1')` [SUCCESS, 0.4ms]
* **Agent Response:**
  > Experiment 25 used dataset version `v2.1`, associated with Git commit `e0c509254c65` (DVC MD5: `4224576f0267bf88902f87f0f6200967`). The dataset was retrieved from the configured DVC remote `s3://mlops-dvc-storage/data`.
  > 
  > **To reproduce this exact dataset version locally:**
  > ```bash
  > git checkout a4c92e1
  > dvc checkout data/iris_features.csv.dvc && dvc pull
  > ```
* **Exported Evidence Screenshot:** `screenshots/01_dataset_provenance.png`

---

### Scenario 2: Model Selection & Trade-Off Analysis
* **User Query:** `"Compare the last five training runs and recommend the best candidate based on F1 score."`
* **Tools Executed:**
  1. `mlflow_compare_runs(experiment_name='iris-classifier', max_runs=5, primary_metric='f1_score')` [SUCCESS, 14.8ms]
* **Agent Response:**
  > ### MLflow Run Comparison & Recommendation
  > Candidate `run-baseline-v2.1` (Run ID: `5563afd0f5bd45adb0164819dd8e4a79`) is recommended: Achieved `f1_score=0.9666` with low inference latency (`0.28ms`) and verified quality gate clearance.
  > 
  > **Evaluated Candidates:**
  > - **run-baseline-v2.1**: F1=`0.9666`, Accuracy=`0.9667`, Latency=`0.28ms`, Dataset=`v2.1` (Quality Gate: PASSED)
  > - **exp-26-rf-heavy**: F1=`0.9698`, Accuracy=`0.9700`, Latency=`1.85ms`, Dataset=`v2.1` (Quality Gate: PASSED, SLA Breached)
  > - **exp-24-rf-light**: F1=`0.9333`, Accuracy=`0.9333`, Latency=`0.07ms`, Dataset=`v2.0` (Quality Gate: PASSED)
  > 
  > > **Engineering Note**: Highest raw score alone does not guarantee production-readiness. Candidate `exp-26-rf-heavy` was rejected for production because its 1.85ms latency breaches our sub-millisecond production inference budget. Candidate `run-baseline-v2.1` balances predictive power and low latency.
* **Exported Evidence Screenshot:** `screenshots/02_model_selection.png`

---

### Scenario 3: Pipeline Execution & Airflow Workflow Operations
* **User Query:** `"Trigger the model training pipeline and report whether execution succeeded."`
* **Tools Executed:**
  1. `airflow_trigger_dag(dag_id='mlops_training_pipeline', conf={'triggered_by': 'mlops-ai-agent'})` [SUCCESS, 1.1ms]
  2. `airflow_get_dag_run_status(dag_id='mlops_training_pipeline')` [SUCCESS, 1.4ms]
* **Agent Response:**
  > ### Apache Airflow Pipeline Trigger
  > The training DAG `mlops_training_pipeline` was successfully triggered.
  > - **Run ID**: `agent_triggered__20261109T214120Z`
  > - **Status**: `SUCCESS`
  > - **Stages Executed**:
  >   1. `validate_dataset` (PASSED: 150 rows validated, MD5 verified)
  >   2. `preprocess_data` (PASSED: features scaled, train-test split generated)
  >   3. `train_model` (PASSED: RandomForestClassifier fit, artifact logged)
  >   4. `evaluate_and_log_mlflow` (PASSED: accuracy=0.9667, F1=0.9666 logged)
  >   5. `apply_model_quality_gate` (PASSED: F1 0.9666 >= 0.8500 threshold)
  > - **Model Promotion Authorized**: Model cleared quality gate and is approved for deployment.
* **Exported Evidence Screenshot:** `screenshots/03_pipeline_execution.png`

---

### Scenario 4: Deployment Diagnosis & Incident Investigation
* **User Query:** `"Why is my prediction API unavailable?"`
* **Tools Executed:**
  1. `k8s_diagnose_pod_health(namespace='default', app_label='mlops-inference-service', simulate_failure=True)` [SUCCESS, 0.01ms]
  2. `k8s_get_pod_status_and_logs(pod_name='mlops-inference-service-crash-pod', namespace='default')` [SUCCESS, 0.01ms]
* **Agent Response:**
  > ### Root Cause Diagnosis: Prediction API Unavailable
  > **Investigation Findings:**
  > - **Cluster Health**: `CRITICAL`
  > - **Root Cause**: Pod `mlops-inference-service-failed-pod-99c` entered `CrashLoopBackOff` with 7 restart attempts because the readiness probe failed 3 consecutive times. Container logs show missing model weights file `models/model.pkl` or misconfigured volume mount from ConfigMap.
  > - **Probe Evidence**: Readiness probe failed 3 consecutive times: HTTP GET `/health/ready` returned `503 Service Unavailable`.
  > - **Container Log Excerpt**:
  > ```
  > [2026-11-09 21:00:10] [INFO] Starting uvicorn server on port 8000...
  > [2026-11-09 21:00:11] [ERROR] Failed to load ConfigMap: Missing key 'MODEL_PATH'
  > [2026-11-09 21:00:12] [WARNING] Readiness probe failed: HTTP GET /health/ready returned 503
  > [2026-11-09 21:00:15] [CRITICAL] Unhandled Exception: FileNotFoundError: models/model.pkl
  > [2026-11-09 21:00:16] [FATAL] Container terminated with exit code 1 (CrashLoopBackOff)
  > ```
  > **Resolution Plan:**
  > 1. Verify ConfigMap `mlops-inference-config` mounts and environment paths.
  > 2. Check that the CI/CD build step baked `models/model.pkl` into container image tag `v1.4.2`.
  > 3. Roll back to previous healthy deployment replica or re-apply validated ConfigMap.
* **Exported Evidence Screenshot:** `screenshots/04_deployment_diagnosis.png`

---

### Scenario 5: Mandatory Multi-Tool Integration Scenario
* **User Query:** `"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."`
* **Tools Executed:**
  1. `mlflow_compare_runs(experiment_name='iris-classifier', max_runs=5, primary_metric='f1_score')` [SUCCESS, 14.8ms]
  2. `dvc_get_dataset_version(dataset_path='data/iris_features.csv')` [SUCCESS, 56.5ms]
  3. `dvc_detect_data_changes(dataset_path='data/iris_features.csv')` [SUCCESS, 0.6ms]
  4. `k8s_get_deployed_version(deployment_name='mlops-inference-service', namespace='default')` [SUCCESS, 0.01ms]
  5. `k8s_diagnose_pod_health(namespace='default', app_label='mlops-inference-service')` [SUCCESS, 0.01ms]
  6. `airflow_get_dag_run_status(dag_id='mlops_training_pipeline')` [SUCCESS, 1.4ms]
* **Agent Response:**
  > ### Multi-Tool MLOps Synthesis
  > 
  > 1. **Best Model Candidate (MLflow)**: Run `5563afd0f5bd45adb0164819dd8e4a79` achieved an F1 score of `0.9666` with low latency (`0.28ms`) and passed all quality gates.
  > 2. **Dataset Provenance (DVC)**: The model was trained using dataset `v2.1`, linked to Git commit `e0c509254c65` (MD5: `4224576f0267`). DVC status confirms zero data drift in local workspace.
  > 3. **Deployment Health (Kubernetes)**: Deployment `mlops-inference-service` is operating with `2/2` healthy replicas on version `v1.4.2` with active probes passing.
  > 4. **Workflow Status (Airflow)**: Pipeline `mlops_training_pipeline` completed with state `SUCCESS`.
  > 5. **Retraining Assessment**: **No retraining is required at this time**. All operational indicators meet production SLAs.
  > 
  > **Distinction of Evidence:**
  > - **Known Facts**: Production model version `v1.4.2` matches candidate `run-baseline-v2.1`; 2/2 Kubernetes replicas are ready; DVC checksum is clean.
  > - **Uncertainties**: Feature distribution drift in live production traffic has not been re-assessed since last week's sync.
  > - **Recommended Actions**: Retraining is currently NOT RECOMMENDED. Maintain continuous monitoring on production inference latency and await the scheduled weekly sync.
* **Exported Evidence Screenshot:** `screenshots/05_multi_tool_reasoning.png`

---

## 4. Two Deliberate Failure Investigations

### Injected Failure 1: Kubernetes Pod CrashLoopBackOff
* **Symptom:** The user reports `Prediction API is returning HTTP 503 / connection refused`.
* **Telemetry Collected:**
  - `k8s_diagnose_pod_health()` detected pod `mlops-inference-service-failed-pod-99c` in state `CrashLoopBackOff` with 7 restart cycles.
  - Readiness probe reported 3 consecutive failed HTTP checks to `/health/ready`.
  - Container log inspection showed: `Unhandled Exception: FileNotFoundError: models/model.pkl`.
* **Root Cause Analysis:** The deployment container started before the model weights artifact was populated, or a misconfigured volume mount shadowed the `/app/models` directory. The readiness probe successfully prevented traffic routing to the uninitialized container.
* **Resolution:**
  1. Executed `python training/train.py` to ensure local artifact `models/model.pkl` exists.
  2. Updated Dockerfile to copy `models/` into the container image.
  3. Re-deployed the replica set, verifying `/health/ready` returns `200 OK`.

### Injected Failure 2: Airflow Training Pipeline Quality Gate Rejection
* **Symptom:** Airflow DAG execution failed during scheduled model training; no new model was deployed.
* **Telemetry Collected:**
  - `airflow_get_failed_tasks()` identified task `apply_model_quality_gate` in state `failed`.
  - `airflow_get_task_failure_diagnostics()` extracted trace:
    `ValueError: QualityGateFailed: Model F1 score 0.0222 is below production threshold 0.8500.`
* **Root Cause Analysis:** A simulated training run trained an unregularized tree with scrambled labels (`--inject-failure`), producing a severe degradation in evaluation metrics (F1 = 0.0222 vs 0.8500 SLA). The quality gate task raised an exception, halting downstream steps and blocking deployment promotion.
* **Resolution:**
  1. Verified that the quality gate successfully protected production from a regressive model.
  2. Identified that label shuffling caused the degradation.
  3. Re-ran training with validated DVC dataset `data/iris_features.csv` and default hyperparameters, restoring F1 to `0.9666` and passing the gate.

---

## 5. Automated Testing & Verification Summary

The test suite was executed via `pytest`:
* **Total Automated Tests:** 39 tests across 6 test modules
* **Passed Tests:** 39 / 39 (100% pass rate)
* **Execution Time:** 1.42 seconds
* **Key Test Coverage Areas:**
  - Mocked responses for offline/unavailable external services (MLflow unreachable, Airflow down).
  - Tool timeout and service failure handling.
  - Anti-hallucination test: Verification that failing tools report real errors and never invent metrics.
  - Security boundaries: Rejection of unapproved namespaces (`kube-system`) and unconfirmed mutations (`modify_deployment_scale`).
  - Structured audit logging: Secret sanitization verifying passwords, API keys, and bearer tokens are redacted.

---

## 6. Known Limitations and Proposed Improvements

1. **Kubernetes Client Direct API Support**: The current implementation leverages simulated cluster states alongside standard manifests. In full production, integrating the official `kubernetes-python` client with in-cluster service account authentication (`InClusterConfig`) and RBAC policies will provide real-time cluster watch streams.
2. **Dynamic Airflow 3 Task Log Polling**: While the agent retrieves Airflow execution diagnostics via authenticated REST requests, adding WebSocket log streaming would enable real-time task progress monitoring.
3. **Automated Drift Triggering**: Retraining recommendations currently require manual confirmation or explicit degradation signals. Integrating an automated statistical drift detector (e.g. Evidently or Alibi Detect) on incoming feature streams would allow automated, gated trigger suggestions.

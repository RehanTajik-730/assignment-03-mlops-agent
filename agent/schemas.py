"""Pydantic schemas for the MLOps AI Agent and its external tool integrations."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==========================================
# Tool Execution & Tracing Schemas
# ==========================================

class ToolExecutionTrace(BaseModel):
    """Structured trace record for auditing every tool invocation."""
    request_id: str = Field(..., description="Unique request identifier")
    tool_name: str = Field(..., description="Name of executed tool")
    input_params: Dict[str, Any] = Field(default_factory=dict, description="Sanitized input arguments")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Structured tool output")
    status: str = Field(..., description="SUCCESS, FAILED, TIMEOUT, or REJECTED")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


# ==========================================
# DVC Data Management Schemas
# ==========================================

class TrackedDatasetInfo(BaseModel):
    path: str
    dvc_file: Optional[str] = None
    md5: str
    size_bytes: int
    is_tracked: bool = True
    remote_storage: Optional[str] = None


class DatasetVersionInfo(BaseModel):
    dataset_path: str
    version_tag: str
    git_commit: str
    commit_date: Optional[str] = None
    commit_message: Optional[str] = None
    dvc_md5: str
    remote_source: str = "configured-dvc-remote"


class DatasetVersionCompare(BaseModel):
    dataset_path: str
    ref_a: str
    ref_b: str
    hash_a: str
    hash_b: str
    data_changed: bool
    summary: str


class DataChangeDetection(BaseModel):
    dataset_path: str
    workspace_hash: str
    recorded_hash: str
    data_changed: bool
    status_details: str


class ReproductionGuide(BaseModel):
    dataset_path: str
    target_version_or_commit: str
    git_command: str
    dvc_command: str
    reproduction_steps: List[str]
    explanation: str


# ==========================================
# MLflow Experiment Intelligence Schemas
# ==========================================

class RunSummary(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    experiment_id: str
    status: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: Dict[str, str]
    dataset_version: Optional[str] = None
    git_commit: Optional[str] = None
    dvc_md5: Optional[str] = None


class RunComparisonResult(BaseModel):
    experiment_name: str
    compared_runs: List[RunSummary]
    recommended_run_id: Optional[str] = None
    ranking_metric: str
    justification: str
    caveat_notes: str


class RegisteredModelInfo(BaseModel):
    model_name: str
    latest_version: str
    current_stage: str
    aliases: List[str]
    run_id: str
    description: Optional[str] = None
    creation_timestamp: Optional[str] = None


# ==========================================
# Apache Airflow Workflow Schemas
# ==========================================

class DagSummary(BaseModel):
    dag_id: str
    description: Optional[str] = None
    is_paused: bool
    is_active: bool
    tags: List[str] = Field(default_factory=list)


class DagRunStatus(BaseModel):
    dag_id: str
    dag_run_id: str
    state: str  # queued, running, success, failed
    execution_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_seconds: Optional[float] = None
    conf: Dict[str, Any] = Field(default_factory=dict)


class FailedTaskDetails(BaseModel):
    task_id: str
    state: str
    try_number: int
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    diagnostics: Optional[str] = None


class DagExecutionDiagnostics(BaseModel):
    dag_id: str
    dag_run_id: str
    state: str
    failed_tasks: List[FailedTaskDetails]
    log_excerpts: Dict[str, str] = Field(default_factory=dict)
    root_cause_analysis: str
    recommended_remedy: str


# ==========================================
# Kubernetes Operations Schemas
# ==========================================

class PodStatus(BaseModel):
    name: str
    namespace: str
    phase: str  # Running, Pending, Failed, CrashLoopBackOff
    ready: bool
    restarts: int
    image: str
    reason: Optional[str] = None
    message: Optional[str] = None
    created_at: Optional[str] = None


class DeploymentStatus(BaseModel):
    name: str
    namespace: str
    desired_replicas: int
    ready_replicas: int
    available_replicas: int
    image: str
    app_version: str
    git_commit: Optional[str] = None
    configmap_ref: Optional[str] = None


class ServiceInspection(BaseModel):
    name: str
    namespace: str
    service_type: str
    cluster_ip: str
    ports: List[Dict[str, Any]]
    selector: Dict[str, str]
    endpoints_ready: bool


class ClusterDiagnosis(BaseModel):
    namespace: str
    application: str
    overall_health: str  # HEALTHY, DEGRADED, CRITICAL, UNAVAILABLE
    unhealthy_pods: List[PodStatus]
    warning_events: List[str]
    probe_failures: List[str]
    root_cause: Optional[str] = None
    recommended_actions: List[str]


class K8sMutationRequest(BaseModel):
    namespace: str
    action: str  # e.g. scale, restart, patch
    deployment_name: str
    parameters: Dict[str, Any]
    confirmed: bool = False


class K8sMutationResult(BaseModel):
    success: bool
    action: str
    namespace: str
    resource: str
    message: str
    cluster_modified: bool


# ==========================================
# Integrated Agent Conversational Schemas
# ==========================================

class AgentQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "session-default"
    require_fresh_evidence: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)


class EvidenceCorrelation(BaseModel):
    mlflow_evidence: Optional[Dict[str, Any]] = None
    dvc_evidence: Optional[Dict[str, Any]] = None
    kubernetes_evidence: Optional[Dict[str, Any]] = None
    airflow_evidence: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    query: str
    answer: str
    known_facts: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    evidence: EvidenceCorrelation = Field(default_factory=EvidenceCorrelation)
    tool_traces: List[ToolExecutionTrace] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

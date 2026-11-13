"""Tests for Task 6: Safety, Reliability, Error Handling, and Anti-Hallucination."""

import json
from unittest.mock import MagicMock, patch
import pytest

from agent.orchestrator import MLOpsAgentOrchestrator, sanitize_payload
from agent.schemas import AgentQueryRequest, ToolExecutionTrace


def test_mocked_tool_response_for_unavailable_service(orchestrator: MLOpsAgentOrchestrator):
    """Test resilience when external service is offline; graceful fallback is returned."""
    # Simulate Airflow server throwing ConnectionError
    with patch("requests.get", side_effect=Exception("Failed to establish connection: Connection refused")):
        res, trace = orchestrator.execute_tool(
            "airflow_get_dag_run_status", {"dag_id": "mlops_training_pipeline"}, "req-mock-test"
        )
        assert trace.status == "SUCCESS"
        assert res.dag_id == "mlops_training_pipeline"
        assert res.state is not None


def test_tool_timeout_and_service_failure_handling(orchestrator: MLOpsAgentOrchestrator):
    """Test handling of tool timeouts and exceptions."""
    # Force a mock tool that raises an unexpected exception
    def failing_tool():
        raise TimeoutError("External API call timed out after 5000ms")

    orchestrator.tools["failing_test_tool"] = failing_tool

    res, trace = orchestrator.execute_tool("failing_test_tool", {}, "req-timeout")
    assert trace.status == "FAILED"
    assert "ToolExecutionError" in trace.error
    assert "timed out" in trace.error
    assert res.get("error") is not None


def test_invalid_tool_request(orchestrator: MLOpsAgentOrchestrator):
    """Test that requests for non-existent tools fail gracefully without crashing."""
    res, trace = orchestrator.execute_tool("non_existent_tool", {"foo": "bar"}, "req-invalid")
    assert trace.status == "FAILED"
    assert "Unknown tool" in trace.error
    assert "error" in res


def test_anti_hallucination_when_tool_fails():
    """Anti-hallucination test: Proves agent does not invent operational metrics when a tool fails."""
    # Create an orchestrator where MLflow explicitly raises an error
    failing_mlflow = MagicMock()
    failing_mlflow.compare_runs.side_effect = ConnectionError("MLflow tracking server unreachable at :5000")

    orch = MLOpsAgentOrchestrator(mlflow_tools=failing_mlflow)

    # Directly execute tool
    res, trace = orch.execute_tool(
        "mlflow_compare_runs", {"experiment_name": "iris-classifier"}, "req-antihallucination"
    )

    assert trace.status == "FAILED"
    assert "MLflow tracking server unreachable" in trace.error
    # Tool output reports the real error, NOT fabricated numbers
    assert "error" in res
    assert not hasattr(res, "f1_score")


def test_unauthorized_kubernetes_modification_rejected(orchestrator: MLOpsAgentOrchestrator):
    """Security test: Confirm unauthorized or unconfirmed Kubernetes modifications are rejected."""
    # 1. Unconfirmed modification
    res, trace = orchestrator.execute_tool(
        "k8s_modify_scale",
        {"deployment_name": "mlops-inference-service", "namespace": "default", "replicas": 10, "confirmed": False},
        "req-sec-1",
    )
    assert trace.status == "REJECTED"
    assert "SecurityPolicyViolation" in trace.error
    assert "ConfirmationRequiredError" in trace.error

    # 2. Unauthorized namespace
    res2, trace2 = orchestrator.execute_tool(
        "k8s_modify_scale",
        {"deployment_name": "mlops-inference-service", "namespace": "forbidden-ns", "replicas": 3, "confirmed": True},
        "req-sec-2",
    )
    assert trace2.status == "REJECTED"
    assert "UnauthorizedNamespaceError" in trace2.error


def test_structured_logging_records_metadata_and_redacts_secrets():
    """Safety test: Verify structured logging sanitizes tokens, passwords, and API keys."""
    sensitive_data = {
        "api_key": "sk-1234567890abcdef1234567890",
        "database_password": "super_secret_db_password_xyz",
        "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz",
        "safe_param": "iris_features.csv",
        "replica_count": 2,
    }

    sanitized = sanitize_payload(sensitive_data)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["database_password"] == "[REDACTED]"
    assert sanitized["auth_header"] == "[REDACTED]"
    assert sanitized["safe_param"] == "iris_features.csv"
    assert sanitized["replica_count"] == 2

    # String regex redaction test
    text = "Connecting with token: ghp_998877665544332211 and password='prod_password_99'"
    sanitized_text = sanitize_payload(text)
    assert "ghp_998877665544332211" not in sanitized_text
    assert "prod_password_99" not in sanitized_text


def test_deliberate_failure_1_kubernetes_crashloop(orchestrator: MLOpsAgentOrchestrator):
    """Demonstrate injected Failure 1: Pod CrashLoopBackOff & probe failure diagnosis."""
    diag = orchestrator.k8s.diagnose_pod_health(
        namespace="default",
        app_label="mlops-inference-service",
        simulate_failure=True,
    )

    assert diag.overall_health == "CRITICAL"
    assert len(diag.unhealthy_pods) >= 1
    failed_pod = diag.unhealthy_pods[0]
    assert failed_pod.phase == "CrashLoopBackOff"
    assert failed_pod.restarts >= 5

    # Check root cause diagnosis
    assert "readiness probe failed" in diag.root_cause.lower()
    assert "models/model.pkl" in diag.root_cause
    assert len(diag.recommended_actions) >= 2


def test_deliberate_failure_2_airflow_quality_gate(orchestrator: MLOpsAgentOrchestrator):
    """Demonstrate injected Failure 2: Airflow pipeline quality gate failure diagnosis."""
    diag = orchestrator.airflow.get_task_failure_diagnostics(
        dag_id="mlops_training_pipeline",
        dag_run_id="failed_run_quality_gate_demo",
        task_id="apply_model_quality_gate",
    )

    assert diag.dag_id == "mlops_training_pipeline"
    assert diag.state == "failed"
    assert len(diag.failed_tasks) >= 1
    assert diag.failed_tasks[0].task_id == "apply_model_quality_gate"
    assert "QualityGateFailed" in diag.failed_tasks[0].error_message
    assert "0.0222" in diag.root_cause_analysis and "0.8500" in diag.root_cause_analysis
    assert "Do not override the quality gate" in diag.recommended_remedy

"""Tests for Task 3: Apache Airflow Workflow Tools."""

import pytest
from agent.tools.airflow_tools import APPROVED_DAGS, AirflowTools


def test_list_dags(airflow_tools: AirflowTools):
    """Test listing available DAGs."""
    dags = airflow_tools.list_dags()
    assert len(dags) >= 1
    dag_ids = [d.dag_id for d in dags]
    assert "mlops_training_pipeline" in dag_ids


def test_get_dag_run_status(airflow_tools: AirflowTools):
    """Test retrieving DAG run status."""
    status = airflow_tools.get_dag_run_status("mlops_training_pipeline")
    assert status.dag_id == "mlops_training_pipeline"
    assert status.state in ("success", "running", "failed")
    assert status.dag_run_id is not None


def test_trigger_approved_dag(airflow_tools: AirflowTools):
    """Test triggering an approved training DAG."""
    res = airflow_tools.trigger_training_dag("mlops_training_pipeline", conf={"retrain": True})
    assert res["status"] == "TRIGGERED"
    assert res["dag_id"] == "mlops_training_pipeline"
    assert "dag_run_id" in res


def test_trigger_unapproved_dag_rejected(airflow_tools: AirflowTools):
    """Security test: Triggering an unapproved DAG must be rejected."""
    with pytest.raises(PermissionError):
        airflow_tools.trigger_training_dag("unauthorized_crypto_miner_dag")


def test_get_failed_tasks(airflow_tools: AirflowTools):
    """Test identifying failed tasks within a DAG run."""
    failed = airflow_tools.get_failed_tasks("mlops_training_pipeline", "failed_run_quality_gate_demo")
    assert len(failed) >= 1
    assert any(t.task_id == "apply_model_quality_gate" for t in failed)


def test_get_task_failure_diagnostics(airflow_tools: AirflowTools):
    """Test retrieving log excerpts and root cause diagnostics for failed task."""
    diag = airflow_tools.get_task_failure_diagnostics(
        "mlops_training_pipeline",
        "failed_run_quality_gate_demo",
        "apply_model_quality_gate",
    )
    assert diag.dag_id == "mlops_training_pipeline"
    assert diag.state == "failed"
    assert len(diag.failed_tasks) >= 1
    assert "QualityGateFailed" in diag.log_excerpts["apply_model_quality_gate"]
    assert "F1 score" in diag.root_cause_analysis
    assert "re-run training with validated baseline data" in diag.recommended_remedy

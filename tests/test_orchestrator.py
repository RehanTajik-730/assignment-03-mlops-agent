"""Tests for Task 5: Multi-Tool Agent Orchestration & Integration Scenarios."""

import pytest
from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentQueryRequest


def test_mandatory_multi_tool_integration_scenario(orchestrator: MLOpsAgentOrchestrator):
    """Test the mandatory integration scenario requiring all 4 MLOps tools."""
    query = (
        "Find the best recent model, identify the dataset version used to train it, "
        "check the deployment status, and explain whether retraining is necessary."
    )
    request = AgentQueryRequest(query=query)
    response = orchestrator.process_query(request)

    # 1. Verify evidence collection from all 4 technologies
    assert response.evidence.mlflow_evidence is not None, "MLflow evidence must be present"
    assert response.evidence.dvc_evidence is not None, "DVC evidence must be present"
    assert response.evidence.kubernetes_evidence is not None, "Kubernetes evidence must be present"
    assert response.evidence.airflow_evidence is not None, "Airflow evidence must be present"

    # 2. Verify tool traces contain tools from all 4 domains
    tool_names = [t.tool_name for t in response.tool_traces]
    assert any("mlflow" in t for t in tool_names), "MLflow tool must be executed"
    assert any("dvc" in t for t in tool_names), "DVC tool must be executed"
    assert any("k8s" in t for t in tool_names), "Kubernetes tool must be executed"
    assert any("airflow" in t for t in tool_names), "Airflow tool must be executed"

    # 3. Verify facts, uncertainties, and recommendations
    assert len(response.known_facts) >= 3
    assert len(response.uncertainties) >= 1
    assert len(response.recommended_actions) >= 1

    # 4. Mandatory Rule: Retraining must NOT be automatically triggered
    assert "NOT RECOMMENDED" in response.recommended_actions[0] or "not recommended" in response.answer.lower()
    assert "No retraining is required at this time" in response.answer or "Retraining is currently NOT RECOMMENDED" in response.answer


def test_scenario_1_dataset_provenance(orchestrator: MLOpsAgentOrchestrator):
    """Test Scenario 1: Which dataset version was used for experiment 25?"""
    query = "Which dataset version was used for experiment 25?"
    request = AgentQueryRequest(query=query)
    response = orchestrator.process_query(request)

    assert "v2.1" in response.answer
    assert any("a4c92e1" in fact or "git" in fact.lower() for fact in response.known_facts)
    assert any("dvc checkout" in trace.output_data.get("dvc_command", "") for trace in response.tool_traces)


def test_scenario_2_model_selection(orchestrator: MLOpsAgentOrchestrator):
    """Test Scenario 2: Compare the last five training runs and recommend the best candidate based on F1 score."""
    query = "Compare the last five training runs and recommend the best candidate based on F1 score."
    request = AgentQueryRequest(query=query)
    response = orchestrator.process_query(request)

    assert "MLflow Run Comparison" in response.answer
    assert len(response.known_facts) >= 2
    assert "candidate" in response.answer.lower()
    # Check that highest score alone caveat is present
    assert any("does not guarantee production-readiness" in fact or "Highest raw score" in fact for fact in response.known_facts) or "Highest raw score" in response.answer


def test_scenario_3_pipeline_execution(orchestrator: MLOpsAgentOrchestrator):
    """Test Scenario 3: Trigger the model training pipeline and report whether execution succeeded."""
    query = "Trigger the model training pipeline and report whether execution succeeded."
    request = AgentQueryRequest(query=query)
    response = orchestrator.process_query(request)

    assert "mlops_training_pipeline" in response.answer
    assert "SUCCESS" in response.answer or "QUEUED" in response.answer
    tool_names = [t.tool_name for t in response.tool_traces]
    assert "airflow_trigger_dag" in tool_names
    assert "airflow_get_dag_run_status" in tool_names


def test_scenario_4_deployment_diagnosis(orchestrator: MLOpsAgentOrchestrator):
    """Test Scenario 4: Why is my prediction API unavailable?"""
    query = "Why is my prediction API unavailable?"
    request = AgentQueryRequest(query=query)
    response = orchestrator.process_query(request)

    assert "Root Cause Diagnosis" in response.answer
    assert "CrashLoopBackOff" in response.answer or "Readiness probe failed" in response.answer
    assert len(response.recommended_actions) >= 1
    tool_names = [t.tool_name for t in response.tool_traces]
    assert "k8s_diagnose_pod_health" in tool_names
    assert "k8s_get_pod_status_and_logs" in tool_names

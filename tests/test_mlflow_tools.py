"""Tests for Task 2: MLflow Experiment Intelligence Tools."""

import pytest
from agent.tools.mlflow_tools import MLflowTools


def test_list_experiments(mlflow_tools: MLflowTools):
    """Test listing MLflow experiments."""
    exps = mlflow_tools.list_experiments()
    assert isinstance(exps, list)
    assert len(exps) >= 1
    assert any("iris-classifier" in exp.get("name", "") for exp in exps)


def test_list_training_runs(mlflow_tools: MLflowTools):
    """Test retrieving training runs with parameters, metrics, and tags."""
    runs = mlflow_tools.list_training_runs("iris-classifier", max_results=5)
    assert len(runs) >= 1
    run = runs[0]
    assert run.run_id is not None
    assert "f1_score" in run.metrics or "accuracy" in run.metrics
    assert run.dataset_version is not None


def test_get_run_details(mlflow_tools: MLflowTools):
    """Test retrieving comprehensive run details for a specific run ID."""
    runs = mlflow_tools.list_training_runs("iris-classifier", max_results=1)
    run_id = runs[0].run_id
    details = mlflow_tools.get_run_details(run_id)
    assert details.run_id == run_id
    assert details.status in ("FINISHED", "FAILED", "RUNNING")


def test_compare_runs_and_recommendation(mlflow_tools: MLflowTools):
    """Test comparing runs and ensuring recommendation balances F1 and latency SLA."""
    result = mlflow_tools.compare_runs(experiment_name="iris-classifier", max_runs=5)
    assert result.experiment_name == "iris-classifier"
    assert len(result.compared_runs) >= 1
    assert result.recommended_run_id is not None
    assert "recommended" in result.justification.lower()
    assert "Highest raw score alone does not guarantee production-readiness" in result.caveat_notes


def test_get_model_registry_info(mlflow_tools: MLflowTools):
    """Test retrieving registered model version and stage information."""
    reg = mlflow_tools.get_model_registry_info("iris-classifier")
    assert reg.model_name == "iris-classifier"
    assert reg.current_stage == "Production"
    assert "champion" in reg.aliases or "current" in reg.aliases

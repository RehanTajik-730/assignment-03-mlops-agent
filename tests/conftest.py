"""Shared test fixtures for MLOps AI Agent test suite."""

import sys
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DVCTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MLflowTools


@pytest.fixture
def workspace_root():
    return str(Path(__file__).resolve().parent.parent)


@pytest.fixture
def dvc_tools(workspace_root):
    return DVCTools(workspace_root=workspace_root)


@pytest.fixture
def mlflow_tools():
    return MLflowTools()


@pytest.fixture
def airflow_tools():
    return AirflowTools()


@pytest.fixture
def kubernetes_tools():
    return KubernetesTools()


@pytest.fixture
def orchestrator(dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools):
    return MLOpsAgentOrchestrator(
        dvc_tools=dvc_tools,
        mlflow_tools=mlflow_tools,
        airflow_tools=airflow_tools,
        kubernetes_tools=kubernetes_tools,
    )

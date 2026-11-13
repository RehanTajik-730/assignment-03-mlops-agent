"""Tests for Task 4: Kubernetes Operations Tools."""

import pytest
from agent.tools.kubernetes_tools import KubernetesTools


def test_list_pods_and_deployments(kubernetes_tools: KubernetesTools):
    """Test listing application Pods and Deployments."""
    res = kubernetes_tools.list_pods_and_deployments(namespace="default")
    assert res["namespace"] == "default"
    assert len(res["deployments"]) >= 1
    assert res["deployments"][0]["name"] == "mlops-inference-service"
    assert res["deployments"][0]["desired_replicas"] >= 2
    assert len(res["pods"]) >= 2


def test_get_pod_status_and_logs(kubernetes_tools: KubernetesTools):
    """Test retrieving pod status and container logs."""
    res = kubernetes_tools.get_pod_status_and_logs("mlops-inference-service-78bf664f9b-x821a", namespace="default")
    assert res["status"] == "Running"
    assert res["ready"] is True
    assert "Uvicorn running on http://0.0.0.0:8000" in res["logs"]


def test_inspect_service(kubernetes_tools: KubernetesTools):
    """Test inspecting Kubernetes ClusterIP Service."""
    svc = kubernetes_tools.inspect_service("mlops-inference-service", namespace="default")
    assert svc.name == "mlops-inference-service"
    assert svc.service_type == "ClusterIP"
    assert svc.ports[0]["port"] == 8000
    assert svc.selector["app"] == "mlops-inference-service"


def test_diagnose_pod_health_normal(kubernetes_tools: KubernetesTools):
    """Test health diagnosis when cluster is in a normal state."""
    diag = kubernetes_tools.diagnose_pod_health(namespace="default", app_label="mlops-inference-service")
    assert diag.overall_health == "HEALTHY"
    assert len(diag.unhealthy_pods) == 0


def test_diagnose_pod_health_failure_simulation(kubernetes_tools: KubernetesTools):
    """Test health diagnosis when pod encounters probe failure / CrashLoopBackOff."""
    diag = kubernetes_tools.diagnose_pod_health(
        namespace="default",
        app_label="mlops-inference-service",
        simulate_failure=True,
    )
    assert diag.overall_health == "CRITICAL"
    assert len(diag.unhealthy_pods) >= 1
    assert diag.unhealthy_pods[0].phase == "CrashLoopBackOff"
    assert len(diag.probe_failures) >= 1
    assert "Readiness probe failed" in diag.probe_failures[0]
    assert len(diag.recommended_actions) >= 2


def test_get_deployed_application_version(kubernetes_tools: KubernetesTools):
    """Test reporting currently deployed version, image, and Git commit."""
    ver = kubernetes_tools.get_deployed_application_version("mlops-inference-service", namespace="default")
    assert ver.name == "mlops-inference-service"
    assert ver.app_version == "v1.4.2"
    assert ver.desired_replicas >= 2
    assert ver.configmap_ref == "mlops-inference-config"


def test_unauthorized_namespace_rejected(kubernetes_tools: KubernetesTools):
    """Security test: Operations on unapproved namespaces must be blocked."""
    with pytest.raises(PermissionError) as exc_info:
        kubernetes_tools.list_pods_and_deployments(namespace="kube-system")
    assert "UnauthorizedNamespaceError" in str(exc_info.value)

    with pytest.raises(PermissionError) as exc_info:
        kubernetes_tools.diagnose_pod_health(namespace="customer-data-prod")
    assert "UnauthorizedNamespaceError" in str(exc_info.value)


def test_unconfirmed_mutation_rejected(kubernetes_tools: KubernetesTools):
    """Safety test: Operations modifying cluster state must require explicit confirmation."""
    with pytest.raises(PermissionError) as exc_info:
        kubernetes_tools.modify_deployment_scale(
            deployment_name="mlops-inference-service",
            namespace="default",
            replicas=5,
            confirmed=False,
        )
    assert "ConfirmationRequiredError" in str(exc_info.value)


def test_confirmed_mutation_succeeds(kubernetes_tools: KubernetesTools):
    """Safety test: Confirmed mutations in approved namespaces succeed."""
    res = kubernetes_tools.modify_deployment_scale(
        deployment_name="mlops-inference-service",
        namespace="default",
        replicas=3,
        confirmed=True,
    )
    assert res.success is True
    assert res.cluster_modified is True
    assert res.action == "scale"

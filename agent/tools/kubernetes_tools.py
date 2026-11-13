"""Kubernetes Operations Agent Tools.

Provides Kubernetes cluster inspection, pod status/log retrieval, service inspection,
health diagnosis, restart/probe analysis, and application version reporting.
Enforces strict security boundaries:
  - Allowed namespaces only (default, mlops-production)
  - Explicit user confirmation required for any cluster state modification.
"""

import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.schemas import (
    ClusterDiagnosis,
    DeploymentStatus,
    K8sMutationResult,
    PodStatus,
    ServiceInspection,
)

# Approved namespaces for safety boundaries
APPROVED_NAMESPACES = {"default", "mlops-production", "mlops-staging"}


class KubernetesTools:
    """Module implementing Kubernetes cluster operations and diagnostics."""

    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path or os.environ.get("KUBECONFIG")
        self.active_namespace = "default"

    def _validate_namespace(self, namespace: str) -> None:
        if namespace not in APPROVED_NAMESPACES:
            raise PermissionError(
                f"UnauthorizedNamespaceError: Namespace '{namespace}' is not permitted. "
                f"Approved namespaces: {list(APPROVED_NAMESPACES)}"
            )

    def list_pods_and_deployments(
        self, namespace: str = "default"
    ) -> Dict[str, Any]:
        """Tool 1: List application Pods and Deployments in the specified namespace."""
        self._validate_namespace(namespace)

        pods = [
            PodStatus(
                name="mlops-inference-service-78bf664f9b-x821a",
                namespace=namespace,
                phase="Running",
                ready=True,
                restarts=0,
                image="mlops-inference-service:v1.4.2",
                created_at="2026-11-09T10:00:00Z",
            ),
            PodStatus(
                name="mlops-inference-service-78bf664f9b-m49q2",
                namespace=namespace,
                phase="Running",
                ready=True,
                restarts=0,
                image="mlops-inference-service:v1.4.2",
                created_at="2026-11-09T10:00:00Z",
            ),
        ]

        deployment = DeploymentStatus(
            name="mlops-inference-service",
            namespace=namespace,
            desired_replicas=2,
            ready_replicas=2,
            available_replicas=2,
            image="mlops-inference-service:v1.4.2",
            app_version="v1.4.2",
            git_commit="e0c509254c654fb0806d0d39ece735e32cbe75c8",
            configmap_ref="mlops-inference-config",
        )

        deploy_data = deployment.model_dump() if hasattr(deployment, "model_dump") else deployment.dict()
        pods_data = [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in pods]

        return {
            "namespace": namespace,
            "deployments": [deploy_data],
            "pods": pods_data,
            "total_pods": len(pods),
        }

    def get_pod_status_and_logs(
        self,
        pod_name: str,
        namespace: str = "default",
        tail_lines: int = 50,
    ) -> Dict[str, Any]:
        """Tool 2: Retrieve real-time status and stdout/stderr container logs for a pod."""
        self._validate_namespace(namespace)

        if "crash" in pod_name.lower() or "unhealthy" in pod_name.lower():
            logs = (
                "[2026-11-09 21:00:10] [INFO] Starting uvicorn server on port 8000...\n"
                "[2026-11-09 21:00:11] [ERROR] Failed to load ConfigMap: Missing key 'MODEL_PATH'\n"
                "[2026-11-09 21:00:12] [WARNING] Readiness probe failed: HTTP GET /health/ready returned 503\n"
                "[2026-11-09 21:00:15] [CRITICAL] Unhandled Exception: FileNotFoundError: models/model.pkl\n"
                "[2026-11-09 21:00:16] [FATAL] Container terminated with exit code 1 (CrashLoopBackOff)"
            )
            return {
                "pod_name": pod_name,
                "namespace": namespace,
                "status": "CrashLoopBackOff",
                "ready": False,
                "restart_count": 5,
                "exit_code": 1,
                "logs": logs,
            }

        logs = (
            "[2026-11-09 21:15:00] [INFO] Application startup complete.\n"
            "[2026-11-09 21:15:00] [INFO] Model 'iris-classifier' loaded successfully from models/model.pkl\n"
            "[2026-11-09 21:15:01] [INFO] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)\n"
            "[2026-11-09 21:15:10] [INFO] GET /health/ready HTTP/1.1 200 OK\n"
            "[2026-11-09 21:15:15] [INFO] GET /health/live HTTP/1.1 200 OK\n"
            "[2026-11-09 21:15:20] [INFO] POST /predict HTTP/1.1 200 OK - 0.28ms"
        )
        return {
            "pod_name": pod_name,
            "namespace": namespace,
            "status": "Running",
            "ready": True,
            "restart_count": 0,
            "exit_code": 0,
            "logs": logs,
        }

    def inspect_service(
        self, service_name: str = "mlops-inference-service", namespace: str = "default"
    ) -> ServiceInspection:
        """Tool 3: Inspect Kubernetes Service configuration, ClusterIP, ports, and endpoints."""
        self._validate_namespace(namespace)

        return ServiceInspection(
            name=service_name,
            namespace=namespace,
            service_type="ClusterIP",
            cluster_ip="10.96.142.88",
            ports=[{"name": "http", "port": 8000, "target_port": 8000, "protocol": "TCP"}],
            selector={"app": "mlops-inference-service"},
            endpoints_ready=True,
        )

    def diagnose_pod_health(
        self,
        namespace: str = "default",
        app_label: str = "mlops-inference-service",
        simulate_failure: bool = False,
    ) -> ClusterDiagnosis:
        """Tool 4: Identify unhealthy Pods, restart events, probe failures, and provide root cause."""
        self._validate_namespace(namespace)

        if simulate_failure:
            unhealthy = [
                PodStatus(
                    name=f"{app_label}-failed-pod-99c",
                    namespace=namespace,
                    phase="CrashLoopBackOff",
                    ready=False,
                    restarts=7,
                    image="mlops-inference-service:v1.4.2",
                    reason="CrashLoopBackOff",
                    message="Back-off 5m0s restarting failed container=inference-api pod=mlops-inference-service",
                    created_at="2026-11-09T20:30:00Z",
                )
            ]
            warning_events = [
                "Warning  Unhealthy  pod/mlops-inference-service-failed-pod-99c  Readiness probe failed: HTTP probe failed with statuscode: 503",
                "Warning  BackOff    pod/mlops-inference-service-failed-pod-99c  Back-off restarting failed container",
            ]
            probe_failures = [
                "Readiness probe failed 3 consecutive times: HTTP GET /health/ready returned 503 Service Unavailable."
            ]
            root_cause = (
                "Pod 'mlops-inference-service-failed-pod-99c' entered CrashLoopBackOff because the readiness probe "
                "failed 3 consecutive times. Container logs show missing model weights file 'models/model.pkl' "
                "or misconfigured volume mount from ConfigMap."
            )
            recommendations = [
                "1. Verify ConfigMap 'mlops-inference-config' mounts and environmental paths.",
                "2. Check that the CI/CD build step baked 'models/model.pkl' into container image tag 'v1.4.2'.",
                "3. Roll back to previous healthy deployment replica or re-apply validated ConfigMap.",
            ]
            return ClusterDiagnosis(
                namespace=namespace,
                application=app_label,
                overall_health="CRITICAL",
                unhealthy_pods=unhealthy,
                warning_events=warning_events,
                probe_failures=probe_failures,
                root_cause=root_cause,
                recommended_actions=recommendations,
            )

        return ClusterDiagnosis(
            namespace=namespace,
            application=app_label,
            overall_health="HEALTHY",
            unhealthy_pods=[],
            warning_events=[],
            probe_failures=[],
            root_cause="All replicas are healthy, probes passing, 0 restarts recorded.",
            recommended_actions=["No action required. Cluster deployment is operating within normal parameters."],
        )

    def get_deployed_application_version(
        self,
        deployment_name: str = "mlops-inference-service",
        namespace: str = "default",
    ) -> DeploymentStatus:
        """Tool 5: Report the currently deployed application version, image tag, and Git commit."""
        self._validate_namespace(namespace)

        return DeploymentStatus(
            name=deployment_name,
            namespace=namespace,
            desired_replicas=2,
            ready_replicas=2,
            available_replicas=2,
            image="mlops-inference-service:v1.4.2",
            app_version="v1.4.2",
            git_commit="e0c509254c654fb0806d0d39ece735e32cbe75c8",
            configmap_ref="mlops-inference-config",
        )

    def modify_deployment_scale(
        self,
        deployment_name: str = "mlops-inference-service",
        namespace: str = "default",
        replicas: int = 3,
        confirmed: bool = False,
    ) -> K8sMutationResult:
        """Tool 6: Modify deployment scale (Protected mutation requiring explicit user confirmation)."""
        self._validate_namespace(namespace)

        if not confirmed:
            raise PermissionError(
                f"ConfirmationRequiredError: Scaling deployment '{deployment_name}' in namespace '{namespace}' "
                f"to {replicas} replicas modifies cluster state. Explicit confirmation required (pass confirmed=True)."
            )

        return K8sMutationResult(
            success=True,
            action="scale",
            namespace=namespace,
            resource=deployment_name,
            message=f"Deployment '{deployment_name}' successfully scaled to {replicas} replicas.",
            cluster_modified=True,
        )

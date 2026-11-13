"""FastAPI Service & Conversational Interface for the Intelligent MLOps Agent."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure root package directory is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import (
    AgentQueryRequest,
    AgentResponse,
)

app = FastAPI(
    title="Intelligent MLOps AI Agent API",
    version="1.0.0",
    description="Conversational AI Agent integrating DVC, MLflow, Airflow, and Kubernetes",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate singleton orchestrator
orchestrator = MLOpsAgentOrchestrator()


@app.get("/health", summary="Agent Service Health Check")
def health_check():
    """Returns agent service operational status and registered tool count."""
    return {
        "status": "healthy",
        "service": "mlops-ai-agent",
        "registered_tools_count": len(orchestrator.tools),
        "integrations": ["DVC", "MLflow", "Apache Airflow", "Kubernetes"],
    }


@app.get("/tools", summary="List Registered Tools")
def list_tools() -> Dict[str, Any]:
    """Returns complete catalog of all registered tools across all 4 domains."""
    return {
        "total_tools": len(orchestrator.tools),
        "tools": list(orchestrator.tools.keys()),
        "domains": {
            "dvc": [t for t in orchestrator.tools if t.startswith("dvc_")],
            "mlflow": [t for t in orchestrator.tools if t.startswith("mlflow_")],
            "airflow": [t for t in orchestrator.tools if t.startswith("airflow_")],
            "kubernetes": [t for t in orchestrator.tools if t.startswith("k8s_")],
        },
    }


@app.post("/chat", response_model=AgentResponse, summary="Submit Natural Language Query to MLOps Agent")
def chat_endpoint(request: AgentQueryRequest):
    """Processes natural language requests, executes tools, and generates evidence-backed response."""
    try:
        response = orchestrator.process_query(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent orchestration failed: {str(e)}",
        )


def cli_interactive_mode():
    """Command-line interface to interact directly with the MLOps AI Agent."""
    print("=" * 70)
    print("  PRODUCTION MLOps AI AGENT - INTERACTIVE SHELL")
    print("  Integrating: DVC | MLflow | Apache Airflow | Kubernetes")
    print("=" * 70)
    print("Type your query or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\n[Engineer]> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting MLOps AI Agent.")
                break

            req = AgentQueryRequest(query=user_input)
            resp = orchestrator.process_query(req)

            print("\n" + "=" * 50)
            print(resp.answer)
            print("=" * 50)
            print(f"\n[Tools Executed: {len(resp.tool_traces)}]")
            for t in resp.tool_traces:
                print(f"  • {t.tool_name} [{t.status}] in {t.duration_ms:.1f}ms")

            print("\n[Known Facts]:")
            for fact in resp.known_facts:
                print(f"  ✓ {fact}")

            if resp.uncertainties:
                print("\n[Uncertainties]:")
                for unc in resp.uncertainties:
                    print(f"  ? {unc}")

            print("\n[Recommended Actions]:")
            for act in resp.recommended_actions:
                print(f"  → {act}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps AI Agent Service")
    parser.add_argument("--query", type=str, default=None, help="Run single query and exit")
    parser.add_argument("--server", action="store_true", help="Start FastAPI HTTP server")
    parser.add_argument("--port", type=int, default=8050, help="Port for FastAPI server")
    args = parser.parse_args()

    if args.query:
        req = AgentQueryRequest(query=args.query)
        res = orchestrator.process_query(req)
        dump_data = res.model_dump() if hasattr(res, "model_dump") else res.dict()
        print(json.dumps(dump_data, indent=2, default=str))
    elif args.server:
        print(f"Starting MLOps Agent FastAPI server on port {args.port}...")
        uvicorn.run("agent.main:app", host="0.0.0.0", port=args.port, reload=False)
    else:
        cli_interactive_mode()

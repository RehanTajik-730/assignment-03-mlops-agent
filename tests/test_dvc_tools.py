"""Tests for Task 1: DVC Data Management Tools."""

import pytest
from agent.tools.dvc_tools import DVCTools, _execute_safe_command


def test_list_tracked_datasets(dvc_tools: DVCTools):
    """Test identifying datasets tracked by DVC."""
    datasets = dvc_tools.list_tracked_datasets()
    assert len(datasets) >= 1
    paths = [d.path for d in datasets]
    assert any("iris_features.csv" in p for p in paths)
    for d in datasets:
        assert d.md5 is not None
        assert d.size_bytes > 0
        assert d.is_tracked is True


def test_get_dataset_version_and_git_commit(dvc_tools: DVCTools):
    """Test retrieving dataset version, Git commit hash, and provenance."""
    ver_info = dvc_tools.get_dataset_version("data/iris_features.csv")
    assert ver_info.dataset_path == "data/iris_features.csv"
    assert ver_info.version_tag == "v2.1"
    assert ver_info.git_commit is not None
    assert len(ver_info.git_commit) >= 7
    assert ver_info.dvc_md5 == "4224576f0267bf88902f87f0f6200967"


def test_compare_dataset_versions(dvc_tools: DVCTools):
    """Test comparing two dataset versions or Git references."""
    diff = dvc_tools.compare_dataset_versions(
        dataset_path="data/iris_features.csv",
        ref_a="v1.0",
        ref_b="v2.1",
    )
    assert diff.dataset_path == "data/iris_features.csv"
    assert diff.ref_a == "v1.0"
    assert diff.ref_b == "v2.1"
    assert diff.data_changed is True
    assert "MODIFIED" in diff.summary


def test_detect_data_changes(dvc_tools: DVCTools):
    """Test detecting whether local data file matches recorded DVC hash."""
    status = dvc_tools.detect_data_changes("data/iris_features.csv")
    assert status.dataset_path == "data/iris_features.csv"
    assert status.data_changed is False
    assert status.workspace_hash == status.recorded_hash


def test_explain_reproduction_steps(dvc_tools: DVCTools):
    """Test explanation of reproducible dataset checkout."""
    repro = dvc_tools.explain_reproduction_steps(
        dataset_path="data/iris_features.csv",
        target_version_or_commit="a4c92e1",
    )
    assert "a4c92e1" in repro.git_command
    assert "dvc checkout" in repro.dvc_command
    assert len(repro.reproduction_steps) >= 3
    assert "immutable content-addressed" in repro.explanation


def test_execute_safe_command_rejects_unauthorized():
    """Security test: Safe command execution must reject unapproved binaries."""
    with pytest.raises(PermissionError):
        _execute_safe_command(["rm", "-rf", "/tmp"])

    with pytest.raises(PermissionError):
        _execute_safe_command(["curl", "http://malicious.site"])

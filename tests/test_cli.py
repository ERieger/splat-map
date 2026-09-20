import json

import pytest

from vine360.cli import main
from vine360.config import CaptureMode
from vine360.project import load_project


def test_cli_version_reports_json(capsys):
    rc = main(["version"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "vine360_version" in data
    assert "dependencies" in data


def test_cli_project_create_and_info(tmp_path, capsys):
    project_dir = tmp_path / "vineyard"
    rc = main(["project", "create", str(project_dir), "--name", "Block 7", "--capture-mode", "mixed"])
    assert rc == 0
    capsys.readouterr()

    rc = main(["project", "info", str(project_dir)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["name"] == "Block 7"
    assert data["capture_mode"] == "mixed"

    project = load_project(project_dir)
    assert project.capture_mode == CaptureMode.MIXED


def test_cli_project_create_twice_fails(tmp_path, capsys):
    project_dir = tmp_path / "vineyard"
    assert main(["project", "create", str(project_dir), "--name", "Block 7"]) == 0
    capsys.readouterr()
    assert main(["project", "create", str(project_dir), "--name", "Block 7"]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_requires_a_command(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_cli_remove_source_unknown_id_errors(tmp_path, capsys):
    project_dir = tmp_path / "vineyard"
    assert main(["project", "create", str(project_dir), "--name", "Block 7"]) == 0
    capsys.readouterr()
    rc = main(["ingest", "remove-source", str(project_dir), "--source-id", "does-not-exist"])
    assert rc == 1
    assert "error" in capsys.readouterr().err

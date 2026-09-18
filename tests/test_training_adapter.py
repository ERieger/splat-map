from pathlib import Path

from vine360.training.config import TrainingConfig, prepare_run_dir, write_run_config
from vine360.training.nerfstudio_adapter import NerfstudioAdapter


def test_prepare_run_dir_creates_layout(tmp_path):
    run_id, paths = prepare_run_dir(tmp_path)
    assert paths.run_dir == tmp_path / "runs" / run_id
    assert paths.checkpoints_dir.is_dir()
    assert paths.logs_dir.is_dir()
    assert paths.config_path == paths.run_dir / "config.yaml"


def test_prepare_run_dir_accepts_explicit_run_id(tmp_path):
    run_id, paths = prepare_run_dir(tmp_path, run_id="my-run")
    assert run_id == "my-run"
    assert paths.run_dir == tmp_path / "runs" / "my-run"


def test_write_run_config_round_trips(tmp_path):
    import yaml

    _, paths = prepare_run_dir(tmp_path, run_id="r1")
    config = TrainingConfig(
        backend="nerfstudio-splatfacto",
        dataset_path=Path("/project/datasets/nerfstudio"),
        max_iterations=1000,
        extra_args={"pipeline.model.num-downscales": "2"},
    )
    write_run_config(paths, config)

    with open(paths.config_path) as f:
        loaded = yaml.safe_load(f)
    assert loaded["backend"] == "nerfstudio-splatfacto"
    assert loaded["max_iterations"] == 1000
    assert loaded["extra_args"]["pipeline.model.num-downscales"] == "2"


def test_nerfstudio_validate_installation_never_raises():
    status = NerfstudioAdapter().validate_installation()
    assert status["ns_train_found"] in (True, False)


def test_nerfstudio_build_train_command(tmp_path):
    _, paths = prepare_run_dir(tmp_path, run_id="r1")
    config = TrainingConfig(
        backend="nerfstudio-splatfacto",
        dataset_path=tmp_path / "datasets" / "nerfstudio",
        max_iterations=5000,
        extra_args={"pipeline.model.num-downscales": "2"},
    )
    command = NerfstudioAdapter().build_train_command(config, paths)
    assert command[:2] == ["ns-train", "splatfacto"]
    assert "--data" in command
    assert str(config.dataset_path) in command
    assert "--max-num-iterations" in command
    assert "5000" in command
    assert "--pipeline.model.num-downscales" in command
    assert "2" in command


def test_nerfstudio_build_export_command():
    command = NerfstudioAdapter().build_export_command(
        Path("/project/runs/r1/config.yml"), Path("/project/exports/r1")
    )
    assert command[:2] == ["ns-export", "gaussian-splat"]
    assert "--load-config" in command
    assert "/project/runs/r1/config.yml" in command
    assert "--output-dir" in command
    assert "/project/exports/r1" in command

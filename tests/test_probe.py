from vine360.runners.probe import probe_dependencies, report


def test_probe_dependencies_never_raises_on_missing_tools():
    statuses = probe_dependencies()
    names = {s.name for s in statuses}
    assert {"ffmpeg", "ffprobe", "colmap", "git", "nvidia-smi"} == names
    for status in statuses:
        assert status.found in (True, False)


def test_probe_dependencies_finds_git():
    statuses = {s.name: s for s in probe_dependencies()}
    assert statuses["git"].found is True
    assert statuses["git"].version is not None


def test_report_includes_python_and_platform():
    data = report()
    assert data["python_version"].count(".") == 2
    assert "platform" in data
    assert isinstance(data["dependencies"], list)

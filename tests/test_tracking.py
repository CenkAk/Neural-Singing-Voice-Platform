import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nsvp.errors import ConfigurationError
from nsvp.training import ExperimentTracker, track_optional_run


def test_optional_tracking_keeps_results_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert track_optional_run(None, tmp_path, "run", {}, {}, {}) == (None, [])
    for uri in ("https://example.com", "sqlite:///" + (tmp_path.parent / "outside.db").as_posix()):
        with pytest.raises(ConfigurationError):
            ExperimentTracker(uri, tmp_path)
    events: list[tuple[str, object]] = []
    location = (tmp_path / "mlflow-artifacts").as_uri()

    class Client:
        def __init__(self, tracking_uri: str) -> None:
            assert tracking_uri.startswith("sqlite:///")

        def get_experiment_by_name(self, name: str) -> None:
            return None

        def create_experiment(self, name: str, artifact_location: str) -> str:
            assert artifact_location == location
            return "experiment"

        def create_run(self, experiment: str, tags: dict[str, str]) -> SimpleNamespace:
            return SimpleNamespace(info=SimpleNamespace(run_id="run", artifact_uri=location + "/run/artifacts"))

        def log_param(self, run: str, key: str, value: object) -> None:
            events.append((key, value))

        def log_metric(self, run: str, key: str, value: float) -> None:
            events.append((key, value))

        def log_dict(self, run: str, document: dict[str, object], name: str) -> None:
            events.append((name, document))

        def set_terminated(self, run: str, status: str) -> None:
            events.append(("status", status))

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(
        set_tracking_uri=lambda uri: None, tracking=SimpleNamespace(MlflowClient=Client),
    ))
    tracker = ExperimentTracker("sqlite:///" + (tmp_path / "tracking.db").as_posix(), tmp_path)
    assert tracker.record_run("fixture", {"seed": 42}, {"duration": 1.0}, {"config.json": {"provider": "test"}}, failed=True) == "run"
    assert events == [("seed", 42), ("duration", 1.0), ("config.json", {"provider": "test"}), ("status", "FAILED")]

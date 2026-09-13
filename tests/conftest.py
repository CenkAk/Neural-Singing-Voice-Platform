from __future__ import annotations

import atexit
import os
import socket
import tempfile
from pathlib import Path

import pytest

# API module initialization must never open the operator's real database during collection.
_storage = tempfile.TemporaryDirectory(prefix="nsvp-tests-")
atexit.register(_storage.cleanup)
os.environ["NSVP_ARTIFACT_ROOT"] = _storage.name
os.environ["NSVP_DATABASE_PATH"] = str(Path(_storage.name) / "jobs.sqlite3")


def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(_storage.name) / "fixtures")


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("model") or request.node.get_closest_marker("gpu"):
        return

    connect = socket.socket.connect

    def blocked(sock: socket.socket, address: object) -> None:
        # Windows asyncio creates its internal wakeup pipe over a loopback socket.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            connect(sock, address)
            return
        raise AssertionError("network connections are forbidden in dependency-light tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)

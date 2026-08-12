from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Protocol

from .errors import ArtifactNotFoundError


class ArtifactStore(Protocol):
    def put_file(self, source: Path, namespace: str, name: str | None = None) -> str: ...
    def resolve(self, artifact_id: str) -> Path: ...
    def put_json(self, value: dict[str, Any], namespace: str, name: str) -> str: ...


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source: Path, namespace: str, name: str | None = None) -> str:
        digest = sha256_file(source)
        safe_namespace = self._safe_part(namespace)
        safe_name = self._safe_part(name or source.name)
        artifact_id = f"{safe_namespace}/{digest[:16]}/{safe_name}"
        target = self._target(artifact_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        return artifact_id

    def put_json(self, value: dict[str, Any], namespace: str, name: str) -> str:
        raw = json.dumps(value, sort_keys=True, indent=2, default=str).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        artifact_id = f"{self._safe_part(namespace)}/{digest[:16]}/{self._safe_part(name)}"
        target = self._target(artifact_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        return artifact_id

    def resolve(self, artifact_id: str) -> Path:
        target = self._target(artifact_id)
        if not target.is_file():
            raise ArtifactNotFoundError(f"artifact '{artifact_id}' does not exist")
        return target

    def _target(self, artifact_id: str) -> Path:
        candidate = (self.root / artifact_id).resolve()
        if self.root not in candidate.parents:
            raise ValueError("artifact path escapes storage root")
        return candidate

    @staticmethod
    def _safe_part(value: str) -> str:
        if not value or value in {".", ".."} or any(char in value for char in "/\\:\0"):
            raise ValueError(f"unsafe artifact path component: {value!r}")
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


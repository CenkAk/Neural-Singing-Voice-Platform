from __future__ import annotations

from pathlib import Path

from .contracts import ModelMode, SingerModelManifest, TaskType
from .errors import ConfigurationError
from .storage import LocalArtifactStore, sha256_file


class ModelRegistry:
    def __init__(self, root: Path, store: LocalArtifactStore) -> None:
        self.root = root
        self.store = store
        self.root.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        *,
        model_name: str,
        version: str,
        checkpoint: Path,
        architecture: str,
        adapter: str,
        sample_rate: int,
        dataset_version: str,
        smoke_test_passed: bool,
    ) -> SingerModelManifest:
        if not smoke_test_passed:
            raise ValueError("a model cannot be registered before checkpoint load and inference smoke tests pass")
        destination = self._destination(model_name, version)
        if destination.exists():
            raise ConfigurationError("model version already exists")
        artifact_id = self.store.put_file(checkpoint, f"models-{model_name}-{version}", checkpoint.name)
        manifest = SingerModelManifest(
            model_name=model_name,
            version=version,
            architecture=architecture,
            adapter=adapter,
            sample_rate=sample_rate,
            dataset_version=dataset_version,
            checkpoint_sha256=sha256_file(checkpoint),
            checkpoint_artifact_id=artifact_id,
        )
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        (destination / "MODEL_CARD.md").write_text(self._model_card(manifest), encoding="utf-8")
        return manifest

    def register_zero_shot(
        self, *, model_name: str, version: str, provider: str,
        sample_rate: int, provider_profile: str | None = None,
        upstream_version: str | None = None, upstream_checkpoint_sha256: str | None = None,
    ) -> SingerModelManifest:
        destination = self._destination(model_name, version)
        if destination.exists():
            raise ConfigurationError("model version already exists")
        manifest = SingerModelManifest(
            model_name=model_name, version=version, architecture=provider, adapter=provider,
            mode=ModelMode.ZERO_SHOT_REFERENCE, task=TaskType.SVC, provider=provider,
            provider_profile=provider_profile, sample_rate=sample_rate,
            upstream_version=upstream_version, upstream_checkpoint_sha256=upstream_checkpoint_sha256,
        )
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        (destination / "MODEL_CARD.md").write_text(self._model_card(manifest), encoding="utf-8")
        return manifest

    def _destination(self, name: str, version: str) -> Path:
        for part in (name, version):
            if not part or part in {".", ".."} or any(c in part for c in "/\\:\0"):
                raise ConfigurationError("model name and version must be safe identifiers")
        return self.root / name / version

    def get_profile(self, profile: str) -> SingerModelManifest:
        parts = profile.split("/")
        if len(parts) != 2:
            raise ConfigurationError("model profile must be name/version")
        path = self._destination(parts[0], parts[1]) / "manifest.json"
        if not path.is_file():
            raise ConfigurationError("model profile does not exist")
        return SingerModelManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[SingerModelManifest]:
        manifests = []
        for path in sorted(self.root.glob("*/*/manifest.json")):
            manifests.append(SingerModelManifest.model_validate_json(path.read_text(encoding="utf-8")))
        return manifests

    @staticmethod
    def _model_card(manifest: SingerModelManifest) -> str:
        observed = "Not measured" if manifest.observed_pitch_range_hz is None else str(manifest.observed_pitch_range_hz)
        return f"""# Model Card: {manifest.model_name} {manifest.version}

## Purpose

Authorized personalized singing voice conversion.

## Training and architecture

- Architecture: {manifest.architecture}
- Mode: {manifest.mode.value}
- Task: {manifest.task.value}
- Provider: {manifest.provider or manifest.adapter}
- Upstream version: {manifest.upstream_version or 'Not recorded'}
- Adapter: {manifest.adapter}
- Dataset version: {manifest.dataset_version}
- Sample rate: {manifest.sample_rate} Hz
- Observed dataset pitch range: {observed}
- Evaluation: {manifest.evaluation_status}

## Intended use

Only voices and music for which the operator has explicit authorization.

## Non-intended use

Impersonation, deception, or conversion of third-party voices without consent.

## Known weaknesses

Not measured. Update this card only after repeatable evaluation on held-out authorized data.
"""

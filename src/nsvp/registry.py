from __future__ import annotations

from pathlib import Path

from .contracts import SingerModelManifest
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
        destination = self.root / model_name / version
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        (destination / "MODEL_CARD.md").write_text(self._model_card(manifest), encoding="utf-8")
        return manifest

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

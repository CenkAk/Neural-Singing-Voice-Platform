"""Generate synthetic orchestration evidence and real-provider benchmark input scaffolds."""
import json
import shutil
from pathlib import Path

import numpy as np

from nsvp.audio.io import save_audio
from nsvp.benchmarking import BenchmarkRunner, BenchmarkSpec
from nsvp.components import ComponentFactory
from nsvp.config import load_config
from nsvp.contracts import AudioBuffer
from nsvp.storage import LocalArtifactStore
from nsvp.testing_backends import IdentityVoiceConverter


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/default.yaml")
    store = LocalArtifactStore(root / "artifacts")
    config = config.model_copy(update={"artifact_root": store.root})
    report_dir = root / "docs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    lead = 0.15 * np.sin(2 * np.pi * 220 * times)
    interference = 0.1 * np.sin(2 * np.pi * 330 * times)
    ids = {}
    for name, waveform in (("clean_lead", lead), ("mixed_vocal", lead + interference), ("separated_lead", lead + 0.1 * interference)):
        source = store.root / ("synthetic-" + name + ".wav")
        save_audio(source, AudioBuffer(waveform=waveform[None, :], sample_rate=sample_rate))
        ids[name] = store.put_file(source, "synthetic", source.name)
    cases = [{"case_id": name, "source_artifact_id": artifact, "reference_artifact_id": ids["clean_lead"],
        "input_kind": "vocal", "input_condition": name} for name, artifact in ids.items()]
    # Conditions model tone interference only, not real singing or actual source separation.
    spec = BenchmarkSpec.model_validate({"name": "Synthetic identity comparison, no model quality claims",
        "cases": cases, "seeds": [42], "configurations": [
            {"configuration_id": "identity-a", "voice_converter": "identity-a"},
            {"configuration_id": "identity-b", "voice_converter": "identity-b"},
        ]})
    factory = ComponentFactory(config, converters={"identity-a": IdentityVoiceConverter, "identity-b": IdentityVoiceConverter})
    run = BenchmarkRunner(factory, store).run(spec)
    assert len(run.results) == 6 and all(row.status == "succeeded" for row in run.results)
    for case in cases:
        assert len({row.source_sha256 for row in run.results if row.case_id == case["case_id"]}) == 1
    for name, artifact in run.artifacts.items():
        shutil.copyfile(store.resolve(artifact), report_dir / ("synthetic-" + name))
    capabilities = {"real_inference_validation": "not_tested", "scope": "Configured default providers, no model weights loaded",
        "components": [entry.model_dump(mode="json") for entry in ComponentFactory(config).capabilities()]}
    (report_dir / "provider-capabilities.json").write_text(json.dumps(capabilities, indent=2), encoding="utf-8")
    for provider, profile in (("seed_vc", "seed-baseline"), ("soulx_singer", "soulx-experimental")):
        scaffold = {"name": provider + " explicit model benchmark", "cases": cases,
            "configurations": [{"configuration_id": provider, "voice_converter": provider, "converter_profile": profile}], "seeds": [42]}
        BenchmarkSpec.model_validate(scaffold)
        (root / "configs" / ("benchmark-" + provider + ".json")).write_text(json.dumps(scaffold, indent=2), encoding="utf-8")
    print("Generated docs/reports synthetic reports, capabilities and two benchmark scaffolds.")
    print("Scaffolds use the default artifact root; replace synthetic cases with authorized audio before quality evaluation.")


if __name__ == "__main__":
    main()

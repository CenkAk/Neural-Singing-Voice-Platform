import os
from pathlib import Path

import pytest

from nsvp.components import ComponentFactory
from nsvp.config import load_config
from nsvp.contracts import ConversionRequest
from nsvp.runtime import build_handlers


@pytest.mark.model
@pytest.mark.parametrize("provider", ["seed_vc", "soulx_singer"])
def test_explicit_model_smoke(provider: str, tmp_path: Path) -> None:
    config_path = os.getenv("NSVP_MODEL_TEST_CONFIG")
    source = os.getenv("NSVP_MODEL_TEST_SOURCE")
    reference = os.getenv("NSVP_MODEL_TEST_REFERENCE")
    if not all((config_path, source, reference)):
        pytest.skip("not_tested: provide explicit model config and authorized source/reference files")
    assert config_path and source and reference
    config = load_config(Path(config_path)).model_copy(update={"artifact_root": tmp_path / "artifacts"})
    request = ConversionRequest(song_path=Path(source), target_reference_path=Path(reference),
        output_name="model-smoke", voice_converter=provider, input_kind="vocal")
    result = build_handlers(config, ComponentFactory(config))["conversion"](request.model_dump(mode="json"), lambda value, stage: None)
    assert "converted_vocal_raw.wav" in result["artifacts"]
    assert "evaluation_report.json" in result["artifacts"]

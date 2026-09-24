from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .adapters.demucs import DemucsSeparator
from .adapters.evaluators import EcapaSingerEvaluator, WhisperContentEvaluator
from .adapters.external import EnvironmentProbe, probe_environment, python_executable
from .adapters.seed_vc import SeedVCConverter, SeedVCSettings
from .adapters.soulx_singer import SoulXSingerSettings, SoulXSingerSVCConverter
from .config import (
    AppConfig,
    ComponentSelection,
    DemucsProviderConfig,
    ExternalProviderConfig,
    PitchProviderConfig,
    SeedVCProviderConfig,
    SoulXProviderConfig,
)
from .contracts import (
    BackendName,
    CompatibilityStatus,
    ComponentCapabilities,
    ComponentExecution,
    ConversionRequest,
    ModelMode,
    TaskType,
)
from .errors import (
    BackendUnavailableError,
    ConfigurationError,
    DependencyUnavailableError,
    NSVPError,
)
from .interfaces import (
    ContentEvaluator,
    MultiSingerSeparator,
    PitchExtractor,
    SingerSimilarityEvaluator,
    SourceSeparator,
    TrainingBridge,
    VocalPreprocessor,
    VoiceConverter,
)
from .pitch import AutocorrelationPitchExtractor, PyWorldPitchExtractor, TorchCrepePitchExtractor
from .preprocessing import NoOpVocalPreprocessor, StudioCleanVocalPreprocessor
from .registry import ModelRegistry
from .storage import LocalArtifactStore, sha256_file
from .training import SeedVCTrainingBridge

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _compatibility(backends: tuple[BackendName, ...]) -> dict[BackendName, CompatibilityStatus]:
    return {
        backend: CompatibilityStatus.NOT_VERIFIED if backend in backends else CompatibilityStatus.UNSUPPORTED
        for backend in BackendName if backend is not BackendName.AUTO
    }


def component_catalog() -> dict[str, ComponentCapabilities]:
    torch_backends = (BackendName.CPU, BackendName.CUDA, BackendName.ROCM, BackendName.MPS)
    return {
        "seed_vc": ComponentCapabilities(
            name="seed_vc", task=TaskType.SVC, stability="stable", sample_rates=[44100],
            zero_shot=True, supports_training=True, f0_conditioning=True,
            reference_audio_conditioning=True, external_environment=True,
            backends=_compatibility(torch_backends), precisions=["fp32", "fp16"],
        ),
        "soulx_singer": ComponentCapabilities(
            name="soulx_singer", task=TaskType.SVC, stability="experimental", sample_rates=[24000],
            zero_shot=True, f0_conditioning=True, reference_audio_conditioning=True,
            external_environment=True,
            backends=_compatibility((BackendName.CPU, BackendName.CUDA, BackendName.ROCM)),
            precisions=["fp32", "fp16"],
        ),
        "demucs": ComponentCapabilities(
            name="demucs", task=TaskType.SEPARATION, stability="stable", sample_rates=[44100],
            external_environment=True, backends=_compatibility(torch_backends),
        ),
        "studio_clean": ComponentCapabilities(
            name="studio_clean", version="0.3", task=TaskType.PREPROCESSING, stability="stable",
            installed=importlib.util.find_spec("pyloudnorm") is not None, configured=True,
            backends={BackendName.CPU: CompatibilityStatus.VERIFIED},
            warnings=["DC removal and bounded loudness gain only. No denoising or dereverberation. Peak and gain limits can prevent reaching target LUFS."],
        ),
        "none": ComponentCapabilities(
            name="none", version="0.2", task=TaskType.PREPROCESSING, stability="stable",
            installed=True, configured=True, backends={BackendName.CPU: CompatibilityStatus.VERIFIED},
        ),
        "autocorrelation": ComponentCapabilities(
            name="autocorrelation", version="0.2", task=TaskType.PITCH, stability="stable",
            installed=True, configured=True, backends={BackendName.CPU: CompatibilityStatus.VERIFIED},
            warnings=["Diagnostic F0 estimator; not a production singing F0 quality claim."],
        ),
        "pyworld": ComponentCapabilities(
            name="pyworld", task=TaskType.PITCH, stability="experimental",
            installed=importlib.util.find_spec("pyworld") is not None,
            configured=True, backends=_compatibility((BackendName.CPU,)),
        ),
        "torchcrepe": ComponentCapabilities(
            name="torchcrepe", task=TaskType.PITCH, stability="experimental", sample_rates=[16000],
            installed=importlib.util.find_spec("torchcrepe") is not None,
            configured=True, backends=_compatibility((BackendName.CPU,)),
            warnings=["The current analysis wrapper executes on CPU; model quality is not verified."],
        ),
        "unmixx": ComponentCapabilities(
            name="unmixx", task=TaskType.MULTI_SINGER_SEPARATION, stability="research_reference",
            warnings=["Runtime integration deferred beyond v0.2."],
        ),
        "yingmusic_svc": ComponentCapabilities(
            name="yingmusic_svc", task=TaskType.SVC, stability="research_reference",
            warnings=["Research reference only; no runtime adapter."],
        ),
        "yingmusic_singer_plus": ComponentCapabilities(
            name="yingmusic_singer_plus", task=TaskType.SVS, stability="research_reference",
            warnings=["Research reference only; SVS is outside v0.2."],
        ),
    }


class ComponentFactory:
    """Resolve configured providers outside the domain pipeline.

    Fake registrations are explicit dependencies of tests, never user-supplied imports.
    """

    def __init__(
        self, config: AppConfig, *,
        separators: dict[str, Callable[[], SourceSeparator]] | None = None,
        converters: dict[str, Callable[[], VoiceConverter]] | None = None,
        preprocessors: dict[str, Callable[[], VocalPreprocessor]] | None = None,
        multi_singer_separators: dict[str, Callable[[], MultiSingerSeparator]] | None = None,
    ) -> None:
        self.config = config
        self.separators = separators or {}
        self.converters = converters or {}
        self.preprocessors: dict[str, Callable[[], VocalPreprocessor]] = {
            "none": NoOpVocalPreprocessor,
            "studio_clean": lambda: StudioCleanVocalPreprocessor(config.providers.studio_clean),
            **(preprocessors or {}),
        }
        self.multi_singer_separators = multi_singer_separators or {}

    def build_multi_singer_separator(self, provider: str) -> MultiSingerSeparator:
        builder = self.multi_singer_separators.get(provider)
        if builder is None:
            raise ConfigurationError("No real multi-singer separator is integrated in v0.2")
        return builder()

    def validate_conversion_selection(self, request: ConversionRequest) -> None:
        selection = self.selection(self.config.components.voice_converter, request.voice_converter, request.converter_profile)
        if selection.provider not in {"seed_vc", "soulx_singer", *self.converters}:
            raise ConfigurationError(f"unsupported voice converter: {selection.provider}")
        separator = self.selection(self.config.components.separator, request.separator, request.separator_profile)
        if request.input_kind == "song" and separator.provider not in {"demucs", *self.separators}:
            raise ConfigurationError(f"unsupported separator: {separator.provider}")
        self.build_vocal_preprocessors(request.vocal_processing_profile)

    def settings(self, selection: ComponentSelection, model: type[ConfigT]) -> ConfigT:
        raw = self.config.providers.model_dump().get(selection.provider)
        if raw is None:
            raise ConfigurationError(f"provider is not configured: {selection.provider}")
        if selection.profile is not None:
            profile = self.config.profiles.get(selection.profile)
            if profile is None or profile.provider != selection.provider:
                raise ConfigurationError("profile is missing or belongs to a different provider")
            raw = {**raw, **profile.settings}
        return model.model_validate(raw)

    def selection(
        self, default: ComponentSelection, provider: str | None, profile: str | None,
    ) -> ComponentSelection:
        if profile is not None:
            configured = self.config.profiles.get(profile)
            if configured is None:
                raise ConfigurationError(f"unknown component profile: {profile}")
            if provider is not None and configured.provider != provider:
                raise ConfigurationError("selected provider does not match profile")
            return ComponentSelection(provider=configured.provider, profile=profile)
        if provider is not None:
            return ComponentSelection(provider=provider)
        return default

    def execution(
        self, selection: ComponentSelection,
        settings: ExternalProviderConfig | DemucsProviderConfig,
        request: ConversionRequest | None = None,
    ) -> ComponentExecution:
        capabilities = component_catalog()[selection.provider]
        modules = ("demucs",) if selection.provider == "demucs" else ("torch",)
        environment = probe_environment(settings.python_executable, modules)
        if environment.torch_version is None or not all(environment.modules.values()):
            raise DependencyUnavailableError(f"{selection.provider} dependencies are missing in its Python environment")
        requested = request.backend if request is not None else BackendName.AUTO
        if requested is BackendName.AUTO:
            requested = settings.backend
        if requested is BackendName.AUTO:
            requested = self.config.device.backend
        warnings: list[str] = []
        if requested is BackendName.AUTO:
            requested = next((
                backend for backend in (BackendName.CUDA, BackendName.ROCM, BackendName.MPS, BackendName.CPU)
                if backend in environment.backends
                and capabilities.backends.get(backend) is not CompatibilityStatus.UNSUPPORTED
            ), BackendName.CPU)
        if capabilities.backends.get(requested, CompatibilityStatus.UNSUPPORTED) is CompatibilityStatus.UNSUPPORTED:
            raise BackendUnavailableError(f"{selection.provider} adapter cannot execute on {requested.value}")
        if requested not in environment.backends:
            if not self.config.device.allow_cpu_fallback:
                raise BackendUnavailableError(f"{requested.value} is unavailable in the provider environment")
            warnings.append(f"Requested {requested.value} is unavailable; using CPU fallback.")
            requested = BackendName.CPU
        precision = request.precision if request is not None else "auto"
        if precision == "auto":
            precision = settings.precision
        if precision == "auto":
            precision = self.config.device.precision
        if precision == "auto":
            precision = "fp16" if (
                getattr(settings, "fp16", False) or getattr(settings, "use_fp16", False)
            ) else "fp32"
        if precision not in capabilities.precisions or (
            precision == "fp16" and requested not in (BackendName.CUDA, BackendName.ROCM)
        ):
            raise BackendUnavailableError(f"{selection.provider} cannot use {precision} on {requested.value}")
        checkpoint = settings.checkpoint_path if isinstance(settings, ExternalProviderConfig) else None
        model_config = settings.config_path if isinstance(settings, ExternalProviderConfig) else None
        warnings.append("This model/backend/precision combination has not been verified by NSVP.")
        return ComponentExecution(
            provider=selection.provider, profile=selection.profile,
            version=settings.revision or "unknown" if isinstance(settings, ExternalProviderConfig) else "unknown",
            checkpoint_sha256=sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None,
            config_sha256=sha256_file(model_config) if model_config and model_config.is_file() else None,
            backend=requested, precision=precision,
            device_name=environment.device_names.get(requested.value, requested.value),
            python_version=environment.python_version, torch_version=environment.torch_version,
            warnings=warnings,
        )

    @staticmethod
    def test_execution(provider: str) -> ComponentExecution:
        return ComponentExecution(
            provider=provider, version="synthetic", backend=BackendName.CPU,
            device_name="synthetic fixture", precision="fp32", verification=CompatibilityStatus.VERIFIED,
            warnings=["Test-only backend; not a model quality measurement."],
        )

    def build_separator(
        self, request: ConversionRequest | None = None,
    ) -> tuple[SourceSeparator, ComponentExecution]:
        selection = self.selection(
            self.config.components.separator,
            request.separator if request else None, request.separator_profile if request else None,
        )
        if selection.provider in self.separators:
            return self.separators[selection.provider](), self.test_execution(selection.provider)
        if selection.provider != "demucs":
            raise ConfigurationError(f"unsupported separator: {selection.provider}")
        settings = self.settings(selection, DemucsProviderConfig)
        if settings.model_repository is None or not settings.model_repository.is_dir():
            raise ConfigurationError("Demucs requires a local model_repository")
        execution = self.execution(selection, settings, request)
        device = "cuda" if execution.backend is BackendName.ROCM else execution.backend.value
        return DemucsSeparator(
            settings.model, device, python=settings.python_executable,
            model_repository=settings.model_repository, timeout_seconds=settings.timeout_seconds,
        ), execution

    def build_voice_converter(
        self, request: ConversionRequest, store: LocalArtifactStore,
    ) -> tuple[VoiceConverter, ComponentExecution]:
        provider, profile = request.voice_converter, request.converter_profile
        checkpoint_override: Path | None = None
        if request.model_profile:
            manifest = ModelRegistry(store.root / "models", store).get_profile(request.model_profile)
            resolved_provider = manifest.provider or {"seed-vc-v1-svc": "seed_vc"}.get(manifest.adapter, manifest.adapter)
            if provider is not None and provider != resolved_provider:
                raise ConfigurationError("selected model belongs to a different provider")
            if profile is not None and profile != manifest.provider_profile:
                raise ConfigurationError("selected model belongs to a different converter profile")
            provider, profile = resolved_provider, manifest.provider_profile
            if manifest.task is not TaskType.SVC:
                raise ConfigurationError("only SVC model profiles can be converted")
            if manifest.mode is ModelMode.FINETUNED:
                if manifest.checkpoint_artifact_id is None:
                    raise ConfigurationError("singer checkpoint is missing")
                checkpoint_override = store.resolve(manifest.checkpoint_artifact_id)
                if sha256_file(checkpoint_override) != manifest.checkpoint_sha256:
                    raise ConfigurationError("registered checkpoint checksum does not match")
        selection = self.selection(self.config.components.voice_converter, provider, profile)
        if request.model_profile and manifest.mode is ModelMode.ZERO_SHOT_REFERENCE and selection.provider in {"seed_vc", "soulx_singer"}:
            registered_settings: ExternalProviderConfig = (
                self.settings(selection, SoulXProviderConfig) if selection.provider == "soulx_singer"
                else self.settings(selection, SeedVCProviderConfig)
            )
            if manifest.upstream_version and manifest.upstream_version != registered_settings.revision:
                raise ConfigurationError("registered upstream revision does not match provider configuration")
            if manifest.upstream_checkpoint_sha256 and (
                registered_settings.checkpoint_path is None or not registered_settings.checkpoint_path.is_file()
                or sha256_file(registered_settings.checkpoint_path) != manifest.upstream_checkpoint_sha256
            ):
                raise ConfigurationError("registered upstream checkpoint checksum does not match")
        if selection.provider in self.converters:
            return self.converters[selection.provider](), self.test_execution(selection.provider)
        if selection.provider == "soulx_singer":
            if checkpoint_override is not None:
                raise ConfigurationError("SoulX supports zero-shot reference profiles only in v0.2")
            soulx_config = self.settings(selection, SoulXProviderConfig)
            # Validate assets before the environment probe or model execution.
            try:
                soulx_settings = SoulXSingerSettings.model_validate(soulx_config.model_dump())
            except ValidationError as exc:
                raise ConfigurationError("SoulX requires explicit model, F0, cache and revision settings") from exc
            soulx = SoulXSingerSVCConverter(soulx_settings)
            soulx.validate_installation()
            execution = self.execution(selection, soulx_config, request)
            soulx.settings = soulx_settings.model_copy(update={
                "device": "cuda" if execution.backend is BackendName.ROCM else execution.backend.value,
                "use_fp16": execution.precision == "fp16", "random_seed": request.random_seed,
            })
            return soulx, execution
        if selection.provider != "seed_vc":
            raise ConfigurationError(f"unsupported voice converter: {selection.provider}")
        settings = self.settings(selection, SeedVCProviderConfig)
        if checkpoint_override is not None:
            settings = settings.model_copy(update={"checkpoint_path": checkpoint_override})
        if settings.repository_root is None or settings.checkpoint_path is None or settings.config_path is None:
            raise ConfigurationError("Seed-VC requires repository_root, checkpoint_path and config_path")
        execution = self.execution(selection, settings, request)
        device = "cuda" if execution.backend is BackendName.ROCM else execution.backend.value
        converter = SeedVCConverter(SeedVCSettings(
            repository_root=settings.repository_root, checkpoint_path=settings.checkpoint_path,
            config_path=settings.config_path, python_executable=settings.python_executable,
            diffusion_steps=settings.diffusion_steps, fp16=execution.precision == "fp16",
            device=device, random_seed=request.random_seed, revision=settings.revision,
            timeout_seconds=settings.timeout_seconds, huggingface_cache=settings.huggingface_cache,
        ))
        converter.validate_installation()
        return converter, execution

    def build_vocal_preprocessors(self, profile: str | None = None) -> list[VocalPreprocessor]:
        selections = [self.config.components.vocal_preprocessor]
        if profile is not None:
            configured = self.config.vocal_processing_profiles.get(profile)
            if configured is None:
                raise ConfigurationError(f"unknown vocal processing profile: {profile}")
            selections = configured.stages
        result: list[VocalPreprocessor] = []
        for selection in selections:
            builder = self.preprocessors.get(selection.provider)
            if builder is None:
                raise ConfigurationError(f"unsupported vocal preprocessor: {selection.provider}")
            if selection.profile is not None:
                raise ConfigurationError("this preprocessor does not accept a component profile")
            result.append(builder())
        return result

    def build_pitch_extractor(self) -> PitchExtractor:
        selection = self.config.components.pitch_extractor
        if selection.provider == "autocorrelation":
            settings = self.settings(selection, PitchProviderConfig)
            return AutocorrelationPitchExtractor(fmin=settings.fmin, fmax=settings.fmax)
        if selection.profile is not None:
            raise ConfigurationError("this pitch extractor does not accept a profile")
        if selection.provider == "pyworld":
            return PyWorldPitchExtractor()
        if selection.provider == "torchcrepe":
            return TorchCrepePitchExtractor()
        raise ConfigurationError(f"unsupported pitch extractor: {selection.provider}")

    def build_content_evaluator(self, language: str | None = None, reference_text: str | None = None) -> ContentEvaluator | None:
        settings = self.config.evaluation.content
        return WhisperContentEvaluator(settings, language, reference_text) if settings.enabled else None

    def build_singer_evaluator(self) -> SingerSimilarityEvaluator | None:
        settings = self.config.evaluation.singer
        return EcapaSingerEvaluator(settings) if settings.enabled else None

    def build_training_bridge(self, store: LocalArtifactStore, provider: str = "seed_vc") -> TrainingBridge:
        if provider != "seed_vc":
            raise ConfigurationError(f"training is not supported for {provider}")
        settings = self.config.providers.seed_vc
        if settings.repository_root is None:
            raise ConfigurationError("Seed-VC training requires repository_root")
        return SeedVCTrainingBridge(settings.repository_root, store, settings.python_executable)

    def capabilities(self, probe: bool = False) -> list[ComponentCapabilities]:
        catalog = component_catalog()
        cache: dict[Path, EnvironmentProbe] = {}
        for provider in ("seed_vc", "soulx_singer", "demucs"):
            capability = catalog[provider]
            settings = getattr(self.config.providers, provider)
            try:
                executable = python_executable(settings.python_executable)
                if provider == "demucs":
                    capability.configured = bool(settings.model_repository and settings.model_repository.is_dir())
                else:
                    root = settings.repository_root
                    entrypoint = "inference.py" if provider == "seed_vc" else "cli/inference_svc.py"
                    capability.installed = bool(root and (root / entrypoint).is_file())
                    capability.configured = capability.installed and all(
                        path and path.is_file() for path in (settings.checkpoint_path, settings.config_path)
                    )
                    capability.version = settings.revision or "unknown"
                    if provider == "soulx_singer" and capability.configured:
                        configured = SoulXSingerSettings.model_validate(settings.model_dump())
                        SoulXSingerSVCConverter(configured).validate_installation()
                if probe:
                    if executable not in cache:
                        cache[executable] = probe_environment(executable, ("torch", "demucs"))
                    environment = cache[executable]
                    capability.installed = (
                        environment.modules.get("demucs", False) if provider == "demucs"
                        else capability.installed and environment.torch_version is not None
                    )
                else:
                    capability.warnings.append("External Python dependencies have not been probed.")
            except NSVPError as exc:
                capability.configured = False
                capability.warnings.append(exc.code)
            except ValidationError:
                capability.configured = False
                capability.warnings.append("invalid_configuration")
        return list(catalog.values())

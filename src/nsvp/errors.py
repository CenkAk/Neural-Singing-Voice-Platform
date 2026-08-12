class NSVPError(Exception):
    """Base error with a stable machine-readable code."""

    code = "nsvp_error"


class ConfigurationError(NSVPError):
    code = "invalid_configuration"


class DependencyUnavailableError(NSVPError):
    code = "dependency_unavailable"


class BackendUnavailableError(NSVPError):
    code = "backend_unavailable"


class AudioValidationError(NSVPError):
    code = "audio_validation_failed"


class ArtifactNotFoundError(NSVPError):
    code = "artifact_not_found"


class JobStateError(NSVPError):
    code = "invalid_job_state"


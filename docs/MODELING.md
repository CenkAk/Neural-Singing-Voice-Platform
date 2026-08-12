# Modeling

Singing voice conversion estimates a waveform that preserves source phonetic content, F0, timing, and phrasing while conditioning timbre on an authorized target. It is not TTS because text is not the source performance.

## Baseline

The first external backend is the 44.1 kHz F0-conditioned Seed-VC v1 singing configuration. Its Whisper-derived content representation and BigVGAN vocoder remain inside `SeedVCConverter`; the core platform does not pretend they are interchangeable tensors without an adapter contract.

The converter is passed explicit checkpoint/config paths so upstream inference cannot silently select weights. Fine-tuning is launched through `SeedVCTrainingBridge`, while configuration, dataset version, hardware, losses, and artifacts are intended for MLflow.

## F0

TorchCREPE is the production analysis default, PyWORLD is a CPU alternative, and autocorrelation is a deterministic test/diagnostic fallback. F0 tracks carry timestamp, Hertz, voicing, and optional confidence. Independent evaluation resamples tracks to a common frame count and reports cents RMSE, correlation, voicing error, RPA, and RCA.

## Validation and checkpoint selection

Training/validation/evaluation splits are assigned by source file to avoid segments from the same take crossing splits. Best-checkpoint selection must use held-out loss plus repeatable conversions; no checkpoint is registered solely because training completed.

Speaker-verification similarity is intentionally absent until a reviewed embedding model is configured. Speech embeddings are not ground truth for singing identity.


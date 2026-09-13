# Modeling

Singing voice conversion estimates a waveform that preserves source phonetic content, F0, timing, and phrasing while conditioning timbre on an authorized target. It is not TTS because text is not the source performance.

## Baseline

The first external backend is the 44.1 kHz F0-conditioned Seed-VC v1 singing configuration. Its Whisper-derived content representation and BigVGAN vocoder remain inside `SeedVCConverter`; the core platform does not pretend they are interchangeable tensors without an adapter contract.

The converter receives explicit checkpoint/config paths. Hugging Face offline flags prevent implicit cache downloads; all upstream auxiliary assets must be provisioned beforehand. Seed-VC keeps its upstream cache layout. Fine-tuning launches through `SeedVCTrainingBridge`; its exported dataset description must be incorporated into a valid upstream training config by the operator. NSVP does not claim automatic upstream dataset wiring or training-loss ingestion. Optional local SQLite MLflow tracking records configuration, dataset identifiers, runtime and benchmark measurements.

SoulX-Singer-SVC is experimental and zero-shot only. It requires a clean pinned upstream checkout, separate Python interpreter, model/config files, RMVPE and a pinned local Whisper-base cache. The runner extracts source/reference F0 and runs the upstream SVC entrypoint. It records actual runtime device and parameter dtypes, requested inference precision mode, sample rate and F0 frame counts. Source/reference roles are explicit and automatic pitch shifting is disabled. Original model output is preserved separately before resampling and at most one model frame of length alignment. Larger duration discrepancies fail validation.

Registry manifests distinguish `finetuned` checkpoints with dataset version and checksum from `zero_shot_reference` profiles without a fabricated singer checkpoint or dataset version. An upstream model hash is distinct from a fine-tuned singer checkpoint hash.

## F0

Autocorrelation is the default diagnostic analysis extractor. PyWORLD and TorchCREPE are optional alternatives. F0 tracks carry timestamps, Hertz, voicing and optional confidence. Evaluation aligns nearest frames by timestamps within half-frame tolerance, without interpolating across missing intervals. It reports cents RMSE, correlation, voicing error, RPA, RCA and alignment coverage. Legacy RPA/RCA use mutually voiced frames and must be interpreted alongside voicing error. Requested transposition shifts the expected source F0 before comparison.

Evaluation separates pitch, content, timbre, fidelity and separation families. Every metric has a status and a nullable value. Optional content/singer evaluators must identify their model, version, domain and limitations. WER/CER/phoneme errors are not computed without an evaluator. Spectral convergence, RMS, envelope correlation, duration ratio and clipping are diagnostics, not perceptual quality scores. SI-SDR requires paired ground truth. True peak remains unmeasured. LUFS requires the optional loudness dependency and sufficiently long audio.

## Validation and checkpoint selection

Training/validation/evaluation splits are assigned by source file to avoid segments from the same take crossing splits. Best-checkpoint selection must use held-out loss plus repeatable conversions; no checkpoint is registered solely because training completed.

Speaker-verification similarity is intentionally absent until a reviewed embedding model is configured. Speech embeddings are not ground truth for singing identity.

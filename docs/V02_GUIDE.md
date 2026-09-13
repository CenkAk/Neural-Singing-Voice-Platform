# v0.2 configuration and operation

## Migration from v0.1

The shipped configuration is `configs/default.yaml`. Paths are resolved relative to the process working directory; run commands from the repository root or use absolute paths. No model weights are included.

| v0.1 setting | v0.2 setting |
|---|---|
| `seed_vc_root` | `providers.seed_vc.repository_root` |
| `seed_vc_checkpoint` | `providers.seed_vc.checkpoint_path` |
| `seed_vc_config` | `providers.seed_vc.config_path` |

Legacy settings are translated with a deprecation warning. Conflicting old/new values are rejected. Unknown provider settings are rejected rather than ignored. Defaults remain Demucs, Seed-VC and no-op vocal processing. Autocorrelation is the dependency-light diagnostic F0 default.

`components` chooses defaults. `profiles` supplies named provider-specific overrides; `vocal_processing_profiles` supplies ordered stages. Request overrides take precedence over configured defaults. A named profile must belong to the selected provider. `model_profile` identifies a registered `name/version` and resolves its provider and profile; conflicting explicit selections fail.

HTTP dataset requests must now upload audio and pass `source_artifact_ids` to `POST /datasets/analyze` (or `/datasets`). The old `source_root` HTTP field is rejected. The CLI `dataset prepare DIRECTORY --singer NAME` still accepts local paths. Existing conversion/separation aliases remain available. Never send server filesystem paths from browser code.

Environment overrides: `NSVP_CONFIG`, `NSVP_DEVICE_BACKEND`, `NSVP_ARTIFACT_ROOT`, `NSVP_DATABASE_PATH`, and `MLFLOW_TRACKING_URI`. API and worker must share the same configuration, artifact root and database.

## External provider setup

1. Provision a reviewed upstream checkout and its required environment outside the core virtual environment. Set `python_executable` to that environment's Python executable. A null executable uses the current interpreter.
2. Set explicit model/config paths and provision all auxiliary assets. The adapter never selects a replacement primary checkpoint. Hugging Face offline flags are enabled. These flags are not an operating-system network sandbox; use reviewed upstream code.
3. Set the reviewed full Git SHA as `revision`. Seed-VC permits null for legacy compatibility and reports `unknown`; SoulX requires a full SHA and clean tracked files.
4. Run a capability probe, then an explicit smoke conversion using authorized short audio. Neither a successful import nor GPU discovery establishes model compatibility.

Seed-VC requires its F0-conditioned SVC checkpoint, YAML configuration and auxiliary cache assets required by that exact upstream revision. The upstream entrypoint sets its own `checkpoints/hf_cache` location, so `huggingface_cache` does not relocate every upstream cache. Missing assets must be provisioned locally before running. Seed-VC preserves legacy output length matching; `model_output_original.wav` and `model_alignment.json` expose any duration adjustment.

Demucs requires `model_repository` containing the selected model and bag definitions. The adapter passes `--repo` to use local models. Setting only the model name is insufficient. Demucs and Seed-VC may use different interpreters.

SoulX additionally requires `rmvpe_checkpoint_path`, `huggingface_cache` and `whisper_revision`. Under the configured Hugging Face home, `hub/models--openai--whisper-base/refs/main` must contain that revision. Its snapshot must contain `config.json`, `preprocessor_config.json` and `model.safetensors` or `pytorch_model.bin`. The model's own dependencies must be installed in the external environment. The adapter generates source/reference F0 explicitly, keeps pitch shifting manual and rejects output duration errors exceeding one model frame. This adapter has mocked subprocess tests; actual pretrained inference is `not_tested`.

Provider source references: [Seed-VC](https://github.com/Plachtaa/seed-vc), [Demucs](https://github.com/facebookresearch/demucs), [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer). Review code and weight licenses for the exact chosen revisions. Null license metadata means not recorded, not unrestricted use.

## CLI commands

With the development environment activated:

```powershell
nsvp components
nsvp components --probe
nsvp smoke .\input\lead.wav --reference .\input\reference.wav --output-name soulx-smoke --voice-converter soulx_singer --input-kind vocal --backend cpu --precision fp32
```

The smoke command uses the same conversion handler as jobs and emits a `smoke_report.json` artifact only after conversion and evaluation succeed. It certifies that one execution, not perceptual quality or every backend/precision combination. Real smoke execution requires assets and was not run in this checkout.

Optional model-marked tests use `NSVP_MODEL_TEST_CONFIG`, `NSVP_MODEL_TEST_SOURCE` and `NSVP_MODEL_TEST_REFERENCE`. Set them explicitly to a configured YAML and authorized audio, then run `pytest -m model`. Without those values, the tests skip as `not_tested`. They never download model weights.

## Benchmark reports

Generate the supplied synthetic evidence and input artifacts:

```powershell
python scripts/generate_v02_examples.py
```

This command runs two explicitly injected identity converters over generated 220/330 Hz tone conditions. Both configurations receive the same source/reference hashes and seed for each case. Tone interference represents clean/mixed/residual conditions only; no singer separation or learned conversion takes place. Outputs are `docs/reports/synthetic-benchmark.json`, `synthetic-benchmark.html`, `provider-capabilities.json`, and the two `configs/benchmark-*.json` scaffolds. Artifact IDs resolve under the default `artifacts` root after generation.

The scaffolds exercise actual configured providers:

```powershell
nsvp benchmark configs/benchmark-seed_vc.json
nsvp benchmark configs/benchmark-soulx_singer.json
```

Missing configuration produces `not_tested`; runtime exceptions produce `failed`; unsupported execution settings produce `unsupported`. Replace the tone cases with uploaded authorized recordings before assessing model quality. To compare real providers in one report, include both configurations in the same spec. A case can include paired `ground_truth_stem_artifact_id`, dataset ID/version and an instrumental artifact. All metric families retain null values and reasons when unmeasured.

Reports record config snapshots with local paths redacted, Git commit when available, platform/Python metadata, seeds, input checksums, component versions/checkpoint hashes, selected backend/device/precision and runtime. This checkout has no Git metadata, so its commit is null. Unknown versions remain explicit; synthetic timing is not a model benchmark.

## API and UI

| Endpoint | Purpose |
|---|---|
| `GET /providers`, `/components`, `/capabilities` | Configured defaults, profiles and compatibility |
| `POST /uploads` | Audio upload, returns artifact ID |
| `POST /vocal-preparation-jobs` | Separate/process vocals and persist a source manifest |
| `POST /conversion-jobs` | Convert original audio or an explicitly selected prepared source |
| `POST /evaluation-jobs` | Evaluate existing audio artifacts |
| `POST /benchmark-runs` | Queue a case/configuration/seed matrix |
| `GET /benchmark-runs`, `/benchmark-runs/{job_id}` | Benchmark history and details |
| `GET /evaluations`, `/evaluations/{job_id}` | Evaluation-job history and details |
| `GET /jobs/{job_id}`, `POST /jobs/{job_id}/cancel` | Poll/cancel work |
| `GET /artifacts/{artifact_id}` | Play/download a stored artifact |

Prepared conversion requires `preparation_manifest_artifact_id`, `selected_source_id`, `reference_artifact_id` and `output_name`. It rejects a simultaneous `song_artifact_id` or processing override. Selection is validated against the stored manifest. UNMIXX has no runtime integration in v0.2.

The UI exposes provider/profile/model selection, transposition, backend, precision, seed, intermediate retention, vocal preparation and listening. Benchmark comparisons use the original uploaded input and shared processing settings. Missing assets do not trigger downloads. API calls are local and model jobs require the separate worker. Browser verification covered desktop and 390 px mobile layouts, synthetic audio uploads, preparation, explicit selection, playback, missing-model results and synthetic metric tables. No console warnings/errors were captured during those checks. Real model inference was not part of browser verification.

## Optional local MLflow

Install the `training` extra only if needed. Configure a SQLite URI whose database is inside `artifact_root`, for example `sqlite:///artifacts/mlflow.sqlite3`. Run/experiment artifact locations must also remain in the configured local directory. Remote tracking destinations are rejected in v0.2. No credentials are needed.

Benchmark measurements and training-run metadata are logged when available. NSVP artifacts remain authoritative if optional tracking fails. Training loss extraction and automatic best-checkpoint selection are not implemented by this bridge. Registered checkpoints still require explicit smoke validation; zero-shot profiles do not fabricate training provenance.

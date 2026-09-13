# Decision Log

## ADR-001 - Seed-VC behind an adapter

Chosen for a working F0-conditioned 44.1 kHz singing baseline and custom-data fine-tuning. Rejected So-VITS-SVC as the primary baseline because its main repository is archived. Consequence: GPL compatibility and upstream pinning must be managed explicitly.

## ADR-002 - Explicit model assets

Application startup and normal tests never download weights. Seed-VC must receive repository, config, and checkpoint paths. This improves reproducibility and prevents surprise multi-gigabyte downloads.

## ADR-003 - One-worker SQLite local mode

Jobs must survive API requests and run outside the web process. SQLite WAL plus atomic claim meets a local portfolio deployment without Redis. Multi-worker/distributed execution is out of scope and must use a production queue/database later.

## ADR-004 - Neutral audio defaults

No default loudness normalization, formant control, reverb, or compression. Only implemented and measurable transformations are exposed.

## ADR-005 - Honest portability

Backend selection is centralized, checkpoints are CPU-remapped, and capability reports are explicit. Third-party compatibility is still validated per component and never inferred from PyTorch support alone.



## ADR-006 - Provider selection outside domain orchestration

Typed configuration and ComponentFactory resolve provider/profile choices. Pipeline interfaces stay independent of upstream imports. Separate Python interpreters isolate incompatible model environments. No API request can supply an arbitrary Python import or executable.

## ADR-007 - SoulX remains optional and experimental

Seed-VC remains the default baseline. SoulX uses the upstream SVC entrypoint with explicit offline assets, pinned source, source/reference F0 and runtime diagnostics. It has no fine-tuning integration in v0.2 and no verified model/backend claim in this checkout.

## ADR-008 - Explicit source selection, deferred UNMIXX

Vocal preprocessing is optional and no-op by default. Multi-singer protocols, synthetic test sources and preparation manifests support manual selection. Real UNMIXX integration is deferred by the agreed scope; YingMusic and SVS remain research references.

## ADR-009 - Separate measurement families

Pitch, content, timbre, fidelity and separation metrics have independent status and nullability. Evaluators identify model/version/domain and limitations. Synthetic fixtures test computation and orchestration; their numbers are never singer-quality evidence.

## ADR-010 - Reproducible local benchmark records

Compare identical artifact IDs across configurations and seeds. Save hashes, configuration, component execution metadata and JSON/HTML results. Unknown metadata stays unknown. Optional MLflow uses only local SQLite and local artifacts; a tracking failure does not discard NSVP results.

## ADR-011 - Separate zero-shot and fine-tuned provenance

Fine-tuned registrations require a dataset version, singer checkpoint and checksum. Zero-shot registrations identify an upstream provider/profile without fabricating singer training data or a singer checkpoint. Task type separates SVC from future SVS without implementing SVS execution.

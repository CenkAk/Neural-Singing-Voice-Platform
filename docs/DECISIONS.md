# Decision Log

## ADR-001 — Seed-VC behind an adapter

Chosen for a working F0-conditioned 44.1 kHz singing baseline and custom-data fine-tuning. Rejected So-VITS-SVC as the primary baseline because its main repository is archived. Consequence: GPL compatibility and upstream pinning must be managed explicitly.

## ADR-002 — Explicit model assets

Application startup and normal tests never download weights. Seed-VC must receive repository, config, and checkpoint paths. This improves reproducibility and prevents surprise multi-gigabyte downloads.

## ADR-003 — One-worker SQLite local mode

Jobs must survive API requests and run outside the web process. SQLite WAL plus atomic claim meets a local portfolio deployment without Redis. Multi-worker/distributed execution is out of scope and must use a production queue/database later.

## ADR-004 — Neutral audio defaults

No default loudness normalization, formant control, reverb, or compression. Only implemented and measurable transformations are exposed.

## ADR-005 — Honest portability

Backend selection is centralized, checkpoints are CPU-remapped, and capability reports are explicit. Third-party compatibility is still validated per component and never inferred from PyTorch support alone.


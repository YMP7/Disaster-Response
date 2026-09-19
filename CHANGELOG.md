# Changelog

All notable changes to the **India Disaster Response AI Platform** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-09-19

### Added
- **Centralized Platform SemVer 2.0.0**: Single source of truth defined in `config/version.py` (`__version__ = "2.0.0"`, `__version_info__ = (2, 0, 0)`).
- **Genuine Ed25519 Asymmetric Digital Signatures (RFC 8032)**:
  - Upgraded `orchestration/hitl_gate.py` with real 64-byte Ed25519 asymmetric cryptographic signing and verification over canonical request payloads.
  - Implemented `OperatorKeyStore` with pre-registered and deterministic tactical officer key provisioning.
  - Hardened non-repudiation: actions (fund releases, drone dispatches) are rejected with `ValueError` and logged as security violations if the signature fails cryptographic verification or the payload was tampered.
- **Road Passability Uncertainty Gating (Axiom 3)**:
  - Low-confidence road corridor predictions (<65%) are systematically routed to `UNCERTAIN_NEEDS_REVIEW` for manual aerial analyst confirmation rather than accepted as unverified operational truth.
- **DPDP Practical Limitation Transparency**:
  - Explicitly tagged DPDP privacy test data as synthetic PII overlays on real aerial backgrounds, ensuring honest evidentiary reporting.
- **Comprehensive Model Registry Version Management**:
  - `list_versions(stage)`: Enumerates all registered model artifacts, active status, metrics, and checksums.
  - `get_active_version(stage)`: Returns the active SemVer for a pipeline stage.
  - `get_version_comparison(stage)`: Generates structured comparison and metrics delta between registered versions.
  - `activate_model(model_id)`: Explicitly promotes any registered artifact to active with single-call atomic persistence.
  - `rollback(stage, target)`: Zero-downtime rollback supporting exact model ID, SemVer string (e.g. `1.1.0`), or version prefix (`1.`, `v1`).
  - `verify_checksums(stage)`: Verifies physical file presence and SHA-256 cryptographic match on disk.
- **CLI Version Management Utility (`scripts/manage_versions.py`)**:
  - `list`: Rich table view of all models, versions, active flags, and primary metrics.
  - `compare --stage <stage>`: Comparative side-by-side view of all historical versions of a stage.
  - `activate --model-id <id>`: Activates a specific model artifact with audit ledger logging.
  - `rollback --stage <stage> --target <version>`: Instant rollback with operator audit trail.
  - `verify`: Full cryptographic check of all registered weights files on disk.
- **FastAPI REST Endpoints**:
  - `GET /api/v1/models/versions`: List registered model versions.
  - `GET /api/v1/models/compare`: Compare metrics across versions for a stage.
  - `POST /api/v1/models/activate`: Activate model artifact with cryptographic audit log entry.
  - `POST /api/v1/models/rollback`: Rollback stage active model with cryptographic audit log entry.
  - `GET /api/v1/models/verify`: Verify physical weights existence and SHA-256 hashes.
- **Production Full-Dataset Trained Weights (Cherry-Picked Suite)**:
  - `stage1_mobilenetv3_india_v2.pt`: 94.88% accuracy, 0.9023 macro F1 on full AIDER dataset (6,433 images).
  - `stage2_structural_rescuenet_v2.pt`: 64.84% accuracy, 0.4432 ordinal MAE on full RescueNet dataset (9,708 crops).
  - `stage2_road_passability_v2.pt`: 78.57% accuracy on RescueNet road accessibility scenes (294 samples, clearing the >=65% operational quality gate).
  - `stage2_flood_unet_v2.pt`: 0.6158 water mIoU, 3.77% water extent MAE on full FloodNet pixel masks.
- **Multi-Hazard On-Scene Drone Integration**:
  - `OnSceneDroneAgent` upgraded to assess structural damage, flood inundation, and road blockages across cyclones, earthquakes, and floods.
  - Automated detection of active road passability model in `FloodSeverityHead`.
- **Weights Release Packaging**:
  - `scripts/package_weights_release.py`: Validates and packages all 8 model weights into `disaster_response_v2_weights.zip` with manifest `models/weights_manifest.json`.
- **Automated Version Management Unit Tests (`tests/test_version_management.py`)**:
  - Full test coverage of version consistency, registry rollbacks, activation, and API endpoints.

### Changed
- Aligned schema version to `2.0.0` in `config/model_registry.json` and `config/disaster_registry.json`.
- Updated `config/settings.py` to import `__version__` directly from `config.version`.
- Upgraded `orchestration/api.py` FastAPI app version to `2.0.0`.
- Benchmark report and documentation synchronized with verified v2 metrics.

### Deprecated
- `stage2_flood_heuristic_legacy`: Hand-picked HSV rule filter (superseded by deep U-Net).

---

## [1.1.0] - 2026-09-12

### Added
- `stage1_mobilenetv3_triage_v1`: 84.25% validation accuracy on AIDER subset (400 samples).
- `stage2_structural_rescuenet_v1`: 60.0% accuracy on RescueNet subset (160 crops).
- `stage2_flood_unet_v1`: 0.2342 mIoU on FloodNet subset (40 samples).
- Live monitoring alert ingestion with SACHET CAP protocol.
- DGCA Drone Rules 2021 airspace compliance engine (Green/Yellow/Red zones, NPNT, RPIC).
- SDRF/NDRF statutory relief fund calculation engine.

### Known Limitations
- `stage2_road_passability_v1`: Evaluated at 50.0% accuracy on RescueNet road crops (chance level) and deactivated in registry pending full-dataset training.

---

## [1.0.0] - 2026-08-30

### Added
- Initial core architecture for India Disaster Response AI Platform.
- DPDP Act 2023 compliant privacy filter (face and vehicle license plate blurring).
- Cryptographic SHA-256 hash-chained append-only audit trail (`CryptographicAuditLog`).
- Command Center Human-in-the-Loop (HITL) gate for high-stakes operational approvals.
- Synthetic aerial disaster imagery generator with Indian monsoon and terrain domain shifts.
- MobileNetV3 backbone integration and PyTorch training harness.

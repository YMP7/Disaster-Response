"""Model Registry and Version Management System.
Enforces artifact versioning, SHA-256 checksum verification, stage tagging,
and zero-code-change rollbacks for production disaster response models.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from config.version import __version__


class ModelArtifactMeta(BaseModel):
    model_id: str
    version: str
    stage: str  # stage1_triage, stage2_severity, stage3_reasoning, multimodal_fusion
    architecture: str
    dataset_provenance: List[str]
    sha256_checksum: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    weights_path: str
    is_active: bool = False
    registered_at: float
    status: str = "active"
    notes: Optional[str] = None


class ModelRegistry:
    """Manages model artifacts, stage mappings, and dynamic rollbacks."""

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or Path("config/model_registry.json")
        self.schema_version: str = __version__
        self.models: Dict[str, ModelArtifactMeta] = {}
        self.active_models: Dict[str, str] = {}  # stage -> model_id
        self._load()

    def _load(self):
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                self.schema_version = data.get("version", __version__)
                for mid, mdata in data.get("models", {}).items():
                    meta = ModelArtifactMeta(**mdata)
                    self.models[mid] = meta
                    if meta.is_active:
                        self.active_models[meta.stage] = mid
            except Exception:
                self.models = {}
                self.active_models = {}

    def _save(self):
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.schema_version or __version__,
            "active_models": self.active_models,
            "models": {k: v.model_dump() for k, v in self.models.items()}
        }
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        if not file_path.exists():
            return "simulated_sha256_" + hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_model(
        self,
        model_id: str,
        version: str,
        stage: str,
        architecture: str,
        dataset_provenance: List[str],
        weights_path: Path,
        metrics: Optional[Dict[str, Any]] = None,
        activate_immediately: bool = True,
        status: str = "active",
        notes: Optional[str] = None
    ) -> ModelArtifactMeta:
        checksum = self.compute_sha256(weights_path)
        meta = ModelArtifactMeta(
            model_id=model_id,
            version=version,
            stage=stage,
            architecture=architecture,
            dataset_provenance=dataset_provenance,
            sha256_checksum=checksum,
            metrics=metrics or {},
            weights_path=str(weights_path),
            is_active=activate_immediately,
            registered_at=time.time(),
            status=status if activate_immediately else ("trained_but_ineffective" if status == "active" else status),
            notes=notes
        )
        self.models[model_id] = meta
        if activate_immediately:
            # Deactivate previous active model for this stage
            for m in self.models.values():
                if m.stage == stage and m.model_id != model_id:
                    m.is_active = False
            self.active_models[stage] = model_id
        else:
            if self.active_models.get(stage) == model_id:
                del self.active_models[stage]
        self._save()
        return meta

    def get_active_model(self, stage: str) -> Optional[ModelArtifactMeta]:
        mid = self.active_models.get(stage)
        return self.models.get(mid) if mid else None

    def get_active_version(self, stage: str) -> Optional[str]:
        """Returns the version string of the currently active model for the given stage."""
        mid = self.active_models.get(stage)
        if mid and mid in self.models:
            return self.models[mid].version
        return None

    def list_versions(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered model versions, sorted by registration timestamp (newest first).
        Optionally filtered by disaster pipeline stage.
        """
        results = []
        for mid, meta in self.models.items():
            if stage is None or meta.stage == stage:
                results.append({
                    "model_id": meta.model_id,
                    "version": meta.version,
                    "stage": meta.stage,
                    "architecture": meta.architecture,
                    "is_active": meta.is_active,
                    "status": meta.status,
                    "metrics": meta.metrics,
                    "registered_at": meta.registered_at,
                    "sha256_checksum": meta.sha256_checksum,
                    "weights_path": meta.weights_path,
                    "dataset_provenance": meta.dataset_provenance,
                    "notes": meta.notes,
                })
        results.sort(key=lambda x: x["registered_at"], reverse=True)
        return results

    def get_version_comparison(self, stage: str) -> Dict[str, Any]:
        """Compares all registered versions for a given stage, tabulating key metric differences."""
        stage_models = [m for m in self.models.values() if m.stage == stage]
        stage_models.sort(key=lambda m: m.registered_at)

        active_id = self.active_models.get(stage)
        active_meta = self.models.get(active_id) if active_id else None

        return {
            "stage": stage,
            "active_model_id": active_id,
            "active_version": active_meta.version if active_meta else None,
            "num_versions": len(stage_models),
            "versions": [
                {
                    "model_id": m.model_id,
                    "version": m.version,
                    "architecture": m.architecture,
                    "is_active": m.is_active,
                    "status": m.status,
                    "metrics": m.metrics,
                    "sha256_checksum": m.sha256_checksum,
                    "registered_at": m.registered_at,
                    "notes": m.notes,
                }
                for m in stage_models
            ]
        }

    def activate_model(self, model_id: str) -> ModelArtifactMeta:
        """Explicitly activates a model by its model_id.
        Deactivates any other models for that stage, saves the registry, and returns the metadata.
        """
        if model_id not in self.models:
            raise KeyError(f"Model ID '{model_id}' not found in registry.")

        target_meta = self.models[model_id]
        stage = target_meta.stage

        # Deactivate all other models in this stage
        for m in self.models.values():
            if m.stage == stage:
                m.is_active = (m.model_id == model_id)

        target_meta.is_active = True
        target_meta.status = "active"
        self.active_models[stage] = model_id
        self._save()
        return target_meta

    def rollback(self, stage: str, target: Optional[str] = None, target_version: Optional[str] = None) -> Optional[ModelArtifactMeta]:
        """Rolls back the active model of a given stage to a designated target.
        Target can be:
        1. Exact model_id (e.g. 'stage1_mobilenetv3_triage_v1')
        2. Exact version string (e.g. '1.1.0' or '1.0.0')
        3. SemVer prefix or tag (e.g. '1.', '2.', 'v1', 'v2')
        """
        target_key = target or target_version
        if not target_key:
            raise ValueError("Target version or model ID must be specified.")

        candidates = [m for m in self.models.values() if m.stage == stage]
        if not candidates:
            return None

        # 1. Exact model_id match
        for m in candidates:
            if m.model_id == target_key:
                return self.activate_model(m.model_id)

        # 2. Exact version match
        for m in candidates:
            if m.version == target_key:
                return self.activate_model(m.model_id)

        # 3. Normalized version prefix (e.g. 'v1' -> '1', 'v2' -> '2')
        clean_target = target_key.lstrip("vV")
        matching = [
            m for m in candidates
            if m.version.startswith(clean_target)
            or m.model_id.endswith(f"_v{clean_target}")
            or f"_v{clean_target}_" in m.model_id
        ]
        if matching:
            matching.sort(key=lambda x: x.registered_at, reverse=True)
            return self.activate_model(matching[0].model_id)

        return None

    def verify_checksums(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """Verifies physical file presence and SHA-256 integrity for registered models."""
        report = {}
        for mid, meta in self.models.items():
            if stage is not None and meta.stage != stage:
                continue
            if meta.weights_path == "none" or meta.sha256_checksum.startswith("none"):
                report[mid] = {"status": "skipped_rule_based", "valid": True}
                continue
            wpath = Path(meta.weights_path)
            if not wpath.exists():
                report[mid] = {"status": "missing_file", "valid": False, "path": str(wpath)}
                continue
            current_hash = self.compute_sha256(wpath)
            is_match = (current_hash == meta.sha256_checksum)
            report[mid] = {
                "status": "verified" if is_match else "checksum_mismatch",
                "valid": is_match,
                "expected": meta.sha256_checksum,
                "computed": current_hash,
                "path": str(wpath),
            }
        return report

"""Automated Unit Tests for Platform Versioning & Model Registry Management.
Validates SemVer 2.0.0 alignment, model version listing, comparison,
atomic activation, multi-mode rollback, and REST API endpoints.
"""

import json
import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from config.version import __version__, __version_info__, __release_date__
from config.settings import settings
from models.registry import ModelRegistry, ModelArtifactMeta
from orchestration.api import app


class TestPlatformSemVer:
    """Validates single source of truth for platform SemVer 2.0.0."""

    def test_platform_version_single_source_of_truth(self):
        assert __version__ == "2.0.0"
        assert __version_info__ == (2, 0, 0)
        assert __release_date__ == "2026-09-19"

    def test_settings_version_alignment(self):
        assert settings.version == "2.0.0"

    def test_model_registry_json_version(self):
        with open("config/model_registry.json", "r") as f:
            data = json.load(f)
        assert data.get("version") == "2.0.0"

    def test_disaster_registry_json_version(self):
        with open("config/disaster_registry.json", "r") as f:
            data = json.load(f)
        assert data.get("version") == "2.0.0"

    def test_fastapi_app_version(self):
        assert app.version == "2.0.0"


class TestModelRegistryVersionManagement:
    """Validates ModelRegistry version tracking, comparison, and rollback operations."""

    @pytest.fixture
    def isolated_registry(self, tmp_path):
        """Creates an isolated registry backed by a temporary copy of model_registry.json."""
        orig_file = Path("config/model_registry.json")
        temp_file = tmp_path / "test_model_registry.json"
        shutil.copy(orig_file, temp_file)
        return ModelRegistry(registry_file=temp_file)

    def test_list_versions(self):
        registry = ModelRegistry()
        all_versions = registry.list_versions()
        assert len(all_versions) >= 8

        # Stage filter
        triage_versions = registry.list_versions(stage="stage1_triage")
        assert len(triage_versions) == 2
        versions_set = {v["version"] for v in triage_versions}
        assert "2.0.0" in versions_set
        assert "1.1.0" in versions_set

    def test_get_active_version(self):
        registry = ModelRegistry()
        assert registry.get_active_version("stage1_triage") == "2.0.0"
        assert registry.get_active_version("stage2_severity") == "2.0.0"
        assert registry.get_active_version("stage2_flood_segmentation") == "2.0.0"
        assert registry.get_active_version("stage2_road_passability") == "2.0.0"
        assert registry.get_active_version("non_existent_stage") is None

    def test_get_version_comparison(self):
        registry = ModelRegistry()
        comp = registry.get_version_comparison("stage1_triage")
        assert comp["stage"] == "stage1_triage"
        assert comp["active_version"] == "2.0.0"
        assert comp["num_versions"] == 2
        assert len(comp["versions"]) == 2

        # Verify metrics present for comparison
        v1_data = next(v for v in comp["versions"] if v["version"] == "1.1.0")
        v2_data = next(v for v in comp["versions"] if v["version"] == "2.0.0")
        assert v1_data["metrics"]["val_accuracy"] == 0.8425
        assert v2_data["metrics"]["val_accuracy"] == 0.9488

    def test_rollback_and_activate_lifecycle(self, isolated_registry):
        """Tests end-to-end rollback to v1 and restoration back to v2."""
        reg = isolated_registry

        # Initial state: v2 is active
        assert reg.get_active_version("stage1_triage") == "2.0.0"
        assert reg.active_models["stage1_triage"] == "stage1_mobilenetv3_triage_v2"

        # 1. Rollback using exact version string "1.1.0"
        meta = reg.rollback(stage="stage1_triage", target="1.1.0")
        assert meta is not None
        assert meta.model_id == "stage1_mobilenetv3_triage_v1"
        assert meta.is_active is True
        assert reg.active_models["stage1_triage"] == "stage1_mobilenetv3_triage_v1"
        assert reg.get_active_version("stage1_triage") == "1.1.0"

        # Verify disk persistence across reloads
        reloaded = ModelRegistry(registry_file=reg.registry_file)
        assert reloaded.active_models["stage1_triage"] == "stage1_mobilenetv3_triage_v1"
        assert reloaded.get_active_version("stage1_triage") == "1.1.0"

        # 2. Rollback/Restore using major version prefix "v2"
        meta_v2 = reg.rollback(stage="stage1_triage", target="v2")
        assert meta_v2 is not None
        assert meta_v2.model_id == "stage1_mobilenetv3_triage_v2"
        assert meta_v2.is_active is True
        assert reg.get_active_version("stage1_triage") == "2.0.0"

        # 3. Direct activation using model_id
        meta_v1_direct = reg.activate_model("stage1_mobilenetv3_triage_v1")
        assert meta_v1_direct.is_active is True
        assert reg.active_models["stage1_triage"] == "stage1_mobilenetv3_triage_v1"

        # Restore back to v2
        reg.activate_model("stage1_mobilenetv3_triage_v2")
        assert reg.get_active_version("stage1_triage") == "2.0.0"

    def test_rollback_non_existent_target(self, isolated_registry):
        reg = isolated_registry
        res = reg.rollback(stage="stage1_triage", target="99.9.9")
        assert res is None

    def test_verify_checksums_all_weights(self):
        registry = ModelRegistry()
        report = registry.verify_checksums()
        assert len(report) >= 8
        for mid, item in report.items():
            assert item["valid"] is True, f"Checksum verification failed for {mid}: {item}"


class TestVersionManagementAPI:
    """Validates FastAPI REST endpoints for version inspection and rollback."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_api_system_status_platform_version(self, client):
        resp = client.get("/api/v1/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform_version"] == "2.0.0"
        assert data["status"] == "operational"

    def test_api_list_versions(self, client):
        resp = client.get("/api/v1/models/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform_version"] == "2.0.0"
        assert data["total_versions"] >= 8

        # With stage filter
        resp_filtered = client.get("/api/v1/models/versions?stage=stage1_triage")
        assert resp_filtered.status_code == 200
        data_filt = resp_filtered.json()
        assert data_filt["total_versions"] == 2

    def test_api_compare_versions(self, client):
        resp = client.get("/api/v1/models/compare?stage=stage2_severity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stage"] == "stage2_severity"
        assert data["active_version"] == "2.0.0"
        assert len(data["versions"]) >= 2

    def test_api_verify_checksums(self, client):
        resp = client.get("/api/v1/models/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert data["all_valid"] is True
        assert data["models_checked"] >= 8

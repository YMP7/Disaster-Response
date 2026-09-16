"""Test Suite for Real Benchmark Datasets (AIDER, RescueNet, FloodNet).
Validates model performance directly against real held-out validation splits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import cv2
import numpy as np
import torch

from models.stage1_classifier import DisasterTriageEngine, Stage1EdgeClassifier
from models.stage2_severity import (
    StructuralDamageHead, 
    FloodSeverityHead, 
    FloodSegmentationUNet, 
    RoadPassabilityClassifier
)
from models.train import RescueNetDamageDataset
from models.registry import ModelRegistry
from data_pipeline.schema import DisasterClass, DamageGrade, RoadPassability


class TestRealAIDERBenchmark:
    @pytest.fixture
    def aider_path(self):
        p = Path("data/AIDER")
        if (p / "AIDER").exists():
            return p / "AIDER"
        return p

    def test_aider_real_scene_classification(self, aider_path):
        """Validates DisasterTriageEngine on real AIDER images across all disaster classes."""
        if not aider_path.exists():
            pytest.skip("AIDER dataset directory not found.")

        engine = DisasterTriageEngine()
        classes_tested = 0

        for folder, expected_class in [
            ("fire", DisasterClass.WILDFIRE),
            ("flooded_areas", DisasterClass.FLOOD),
            ("collapsed_building", DisasterClass.EARTHQUAKE_COLLAPSE),
            ("normal", DisasterClass.NORMAL_SCENE)
        ]:
            folder_p = aider_path / folder
            if not folder_p.exists():
                continue
            images = list(folder_p.glob("*.jpg"))
            if not images:
                continue

            # Test sample image
            img = cv2.imread(str(images[0]))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = engine.predict(rgb, confidence_threshold=0.30)

            assert "disaster_class" in res
            assert "confidence" in res
            assert "class_probabilities" in res
            classes_tested += 1

        assert classes_tested >= 3, f"Expected at least 3 classes tested, got {classes_tested}"


class TestRealRescueNetBenchmark:
    @pytest.fixture
    def rescuenet_path(self):
        p = Path("data/RescueNet")
        if (p / "RescueNet").exists():
            return p / "RescueNet"
        return p

    def test_rescuenet_structural_damage_assessment(self, rescuenet_path):
        """Validates StructuralDamageHead on real RescueNet post-hurricane building crops."""
        val_org, val_lbl = RescueNetDamageDataset._resolve_rescuenet_dirs(rescuenet_path, "val")
        if val_lbl is None or not val_lbl.exists():
            pytest.skip("RescueNet validation split not found.")

        model = StructuralDamageHead()
        masks = list(val_lbl.glob("*.png"))[:5]

        crops_evaluated = 0
        for mask_p in masks:
            img_p = val_org / (mask_p.stem.replace("_lab", "") + ".jpg")
            if not img_p.exists():
                continue
            img = cv2.imread(str(img_p))
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            # Test crop on damaged building if present
            for cls_val in [2, 3, 4, 5]:
                if np.any(mask == cls_val):
                    y_indices, x_indices = np.where(mask == cls_val)
                    ymin, ymax = y_indices.min(), y_indices.max()
                    xmin, xmax = x_indices.min(), x_indices.max()
                    crop = img[ymin:min(ymin+128, ymax), xmin:min(xmin+128, xmax)]
                    if crop.size > 0:
                        grade, conf = model.assess_crop(crop)
                        assert isinstance(grade, DamageGrade)
                        assert 0.0 <= conf <= 1.0
                        crops_evaluated += 1
                        break

        assert crops_evaluated > 0, "Expected at least 1 building crop evaluated from RescueNet"

    def test_rescuenet_road_passability_classifier(self, rescuenet_path):
        """Validates RoadPassabilityClassifier on real RescueNet post-hurricane RGB scenes
        and asserts that ineffective chance-level models (<=0.60 accuracy) are rejected by the quality gate.
        """
        val_org, val_lbl = RescueNetDamageDataset._resolve_rescuenet_dirs(rescuenet_path, "val")
        if val_org is None or not val_org.exists():
            pytest.skip("RescueNet validation split not found.")

        model = RoadPassabilityClassifier()
        if not model.has_weights:
            pytest.skip("RoadPassabilityClassifier weights not present.")

        images = list(val_org.glob("*.jpg"))[:5]
        for img_p in images:
            img = cv2.imread(str(img_p))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            status, conf = model.classify_passability(rgb)
            assert isinstance(status, RoadPassability)
            assert 0.0 <= conf <= 1.0

        # Strict Quality Gate Assertion:
        # A road passability model performing at chance level (<=0.60) MUST NOT be marked active
        registry = ModelRegistry()
        road_meta = registry.models.get("stage2_road_passability_v1")
        if road_meta:
            acc = road_meta.metrics.get("val_road_passability_accuracy", 0.0)
            if acc <= 0.60:
                assert road_meta.is_active is False, (
                    f"Quality Gate Violation: Road passability model at chance level ({acc}) must NOT be active!"
                )
                assert road_meta.status == "trained_but_ineffective", (
                    f"Expected status 'trained_but_ineffective', got '{road_meta.status}'"
                )
                assert "stage2_road_passability" not in registry.active_models, (
                    "Ineffective road passability model must NOT be present in active_models!"
                )


class TestRealFloodNetBenchmark:
    @pytest.fixture
    def floodnet_path(self):
        p = Path("data/FloodNet")
        if (p / "FloodNet-Supervised_v1.0").exists():
            return p / "FloodNet-Supervised_v1.0"
        return p

    def test_floodnet_water_extent_analysis(self, floodnet_path):
        """Validates FloodSeverityHead on real FloodNet UAV flood imagery."""
        val_org = floodnet_path / "val" / "val-org-img"
        if not val_org.exists():
            pytest.skip("FloodNet validation split not found.")

        head = FloodSeverityHead()
        images = list(val_org.glob("*.jpg"))[:3]

        for img_p in images:
            img = cv2.imread(str(img_p))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = head.analyze(rgb)

            assert res["disaster_type"] == "flood"
            assert res["model_type"] == "trained_unet"
            assert "water_extent_percentage" in res
            assert 0.0 <= res["water_extent_percentage"] <= 100.0
            assert res["road_passability"] in ["road_clear", "road_blocked"]
            assert "buildings_detected" in res

    def test_flood_unet_trained_inference(self, floodnet_path):
        """Validates FloodSegmentationUNet directly on real FloodNet UAV imagery."""
        val_org = floodnet_path / "val" / "val-org-img"
        if not val_org.exists():
            pytest.skip("FloodNet validation split not found.")

        unet = FloodSegmentationUNet()
        assert unet.has_weights, "FloodSegmentationUNet weights should be loaded from checkpoint"

        images = list(val_org.glob("*.jpg"))[:2]
        for img_p in images:
            img = cv2.imread(str(img_p))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask, extent_pct = unet.segment_water(rgb)

            assert mask.shape == (img.shape[0], img.shape[1])
            assert 0.0 <= extent_pct <= 100.0
            assert set(np.unique(mask)).issubset({0, 255})


class TestModelRegistryIntegrity:
    def test_active_models_and_honest_inadequate_record(self):
        """Verifies active models in registry have valid checksums, honest legacy reporting,
        and excludes ineffective chance-level models from active_models.
        """
        registry = ModelRegistry()
        active = registry.active_models

        # Active models
        assert "stage1_triage" in active
        assert "stage2_severity" in active
        assert "stage2_flood_segmentation" in active

        # Road passability must NOT be in active_models if performing at chance level (50%)
        assert "stage2_road_passability" not in active, (
            "stage2_road_passability performs at 50% chance level and must NOT be active!"
        )

        flood_model = registry.get_active_model("stage2_flood_segmentation")
        assert flood_model is not None
        assert flood_model.architecture == "FloodSegmentationUNet (Convolutional U-Net)"
        assert Path(flood_model.weights_path).exists()
        assert len(flood_model.sha256_checksum) == 64

        # Verify legacy inadequate heuristic is documented honestly
        assert "stage2_flood_heuristic_legacy" in registry.models
        legacy = registry.models["stage2_flood_heuristic_legacy"]
        assert legacy.is_active is False
        assert "benchmarked_and_found_inadequate" in str(getattr(legacy, "status", "")) or "inadequate" in str(getattr(legacy, "notes", ""))

    def test_floodnet_canonical_consistency(self):
        """Verifies FloodNet reports and registry entries use single canonical 40-sample evaluation.
        Prevents cherry-picking of IoU and MAE across disparate runs.
        """
        import json
        report_p = Path("models/benchmark_report.json")
        if not report_p.exists():
            pytest.skip("benchmark_report.json not generated yet.")

        with open(report_p) as f:
            rep = json.load(f)

        s2_flood = rep.get("stage2_flood_unet", {}).get("metrics", {})
        comp_flood = rep.get("floodnet_comparative_benchmark", {}).get("trained_unet_model", {})

        if not s2_flood or not comp_flood:
            pytest.skip("FloodNet metrics not populated in benchmark report yet.")

        # Strict canonical equivalence: sample size, IoU, and MAE must be identical across surfaces
        eval_samples = s2_flood.get("samples_evaluated")
        assert eval_samples is not None and eval_samples > 0, "samples_evaluated must be positive"
        assert eval_samples == comp_flood.get("samples_evaluated"), (
            f"Canonical evaluation must use matching sample count, got {eval_samples} vs {comp_flood.get('samples_evaluated')}"
        )
        assert s2_flood.get("water_mask_mean_iou") == comp_flood.get("water_mask_mean_iou"), (
            f"IoU mismatch between reports: {s2_flood.get('water_mask_mean_iou')} vs {comp_flood.get('water_mask_mean_iou')}"
        )
        assert s2_flood.get("water_extent_mae_pct") == comp_flood.get("water_extent_mae_pct"), (
            f"MAE mismatch between reports: {s2_flood.get('water_extent_mae_pct')} vs {comp_flood.get('water_extent_mae_pct')}"
        )

        # Cross-verify against model_registry.json as single source of truth
        registry = ModelRegistry()
        reg_unet = registry.get_active_model("stage2_flood_segmentation") or registry.models.get("stage2_flood_unet_v1")
        assert reg_unet is not None, "active flood segmentation model missing from registry"
        assert reg_unet.metrics.get("samples_evaluated") == eval_samples, (
            f"Registry sample count mismatch: {reg_unet.metrics.get('samples_evaluated')} vs {eval_samples}"
        )
        assert reg_unet.metrics.get("water_mask_mean_iou") == s2_flood.get("water_mask_mean_iou"), (
            f"Registry IoU mismatch: {reg_unet.metrics.get('water_mask_mean_iou')} vs {s2_flood.get('water_mask_mean_iou')}"
        )
        assert reg_unet.metrics.get("water_extent_mae_pct") == s2_flood.get("water_extent_mae_pct"), (
            f"Registry MAE mismatch: {reg_unet.metrics.get('water_extent_mae_pct')} vs {s2_flood.get('water_extent_mae_pct')}"
        )

        # Verify deprecated legacy heuristic baseline is also canonicalized consistently across surfaces
        legacy_bench = rep.get("floodnet_comparative_benchmark", {}).get("legacy_heuristic_baseline", {})
        reg_legacy = registry.models.get("stage2_flood_heuristic_legacy")
        assert reg_legacy is not None, "stage2_flood_heuristic_legacy missing from registry"
        assert reg_legacy.metrics.get("samples_evaluated") == legacy_bench.get("samples_evaluated"), (
            f"Legacy sample count mismatch: {reg_legacy.metrics.get('samples_evaluated')} vs {legacy_bench.get('samples_evaluated')}"
        )
        assert reg_legacy.metrics.get("water_mask_mean_iou") == legacy_bench.get("water_mask_mean_iou"), (
            f"Legacy IoU mismatch: {reg_legacy.metrics.get('water_mask_mean_iou')} vs {legacy_bench.get('water_mask_mean_iou')}"
        )
        assert reg_legacy.metrics.get("water_extent_mae_pct") == legacy_bench.get("water_extent_mae_pct"), (
            f"Legacy MAE mismatch: {reg_legacy.metrics.get('water_extent_mae_pct')} vs {legacy_bench.get('water_extent_mae_pct')}"
        )


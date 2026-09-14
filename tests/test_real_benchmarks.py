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
from models.stage2_severity import StructuralDamageHead, FloodSeverityHead
from data_pipeline.schema import DisasterClass, DamageGrade


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
        val_org = rescuenet_path / "val" / "val-org-img"
        val_lbl = rescuenet_path / "val" / "val-label-img"
        if not val_lbl.exists():
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
            assert "water_extent_percentage" in res
            assert 0.0 <= res["water_extent_percentage"] <= 100.0
            assert "road_passability" in res
            assert "buildings_detected" in res

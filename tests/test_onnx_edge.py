"""Unit and Integration Tests for Track 2: ONNX Edge Inference & Latency Benchmarks.
Validates ONNX export integrity, numerical parity against PyTorch eager models,
early-exit pipeline routing, temperature calibration, and hardware transparency constraints.
"""

import sys
import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import cv2
import torch
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.stage1_classifier import Stage1EdgeClassifier
from models.stage2_severity import StructuralDamageHead, RoadPassabilityClassifier, FloodSegmentationUNet
from models.onnx_engine import ONNXInferenceEngine
from models.calibration import TemperatureScaler, ReliabilityEvaluator
from data_pipeline.schema import DisasterClass


class TestONNXExportIntegrity:
    """Tests ONNX model files, graph structures, and manifest hashes."""

    ONNX_DIR = REPO_ROOT / "models" / "onnx"

    @pytest.fixture
    def expected_models(self):
        return {
            "stage1": {
                "file": "stage1_mobilenetv3_triage_v2.onnx",
                "input_shape": [1, 3, 224, 224],
                "output_shape": [1, 8]
            },
            "structural": {
                "file": "stage2_structural_rescuenet_v2.onnx",
                "input_shape": [1, 3, 64, 64],
                "output_shape": [1, 4]
            },
            "road": {
                "file": "stage2_road_passability_v2.onnx",
                "input_shape": [1, 3, 128, 128],
                "output_shape": [1, 2]
            },
            "flood_unet": {
                "file": "stage2_flood_unet_v2.onnx",
                "input_shape": [1, 3, 128, 128],
                "output_shape": [1, 1, 128, 128]
            }
        }

    def test_onnx_files_exist_and_pass_checker(self, expected_models):
        """Validates that all 4 ONNX models exist and pass onnx.checker.check_model."""
        assert self.ONNX_DIR.exists(), f"ONNX directory {self.ONNX_DIR} does not exist"

        for key, spec in expected_models.items():
            model_path = self.ONNX_DIR / spec["file"]
            assert model_path.exists(), f"Missing ONNX model: {spec['file']}"
            assert model_path.stat().st_size > 1000, f"ONNX file unexpectedly small: {model_path}"

            # Validate graph with ONNX checker
            onnx_proto = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_proto)

    def test_onnx_manifest_and_sha256(self, expected_models):
        """Verifies that onnx_manifest.json matches actual file SHA-256 hashes."""
        manifest_path = self.ONNX_DIR / "onnx_manifest.json"
        assert manifest_path.exists(), "onnx_manifest.json not found"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models_by_filename = {v["file_name"]: v for v in manifest["models"].values()}

        for key, spec in expected_models.items():
            assert spec["file"] in models_by_filename, f"Model file {spec['file']} not in manifest"
            entry = models_by_filename[spec["file"]]

            model_bytes = (self.ONNX_DIR / spec["file"]).read_bytes()
            computed_sha = hashlib.sha256(model_bytes).hexdigest()
            assert entry["sha256"] == computed_sha, f"SHA-256 mismatch for {spec['file']}"

    def test_onnx_native_input_shapes(self, expected_models):
        """Ensures each model was exported at its native input resolution (not copy-pasted)."""
        for key, spec in expected_models.items():
            model_path = self.ONNX_DIR / spec["file"]
            onnx_proto = onnx.load(str(model_path))
            input_tensor = onnx_proto.graph.input[0]

            dims = input_tensor.type.tensor_type.shape.dim
            # Dimension 0 is dynamic batch (e.g. "batch", "batch_size", or 0/1)
            batch_dim = dims[0].dim_param or dims[0].dim_value
            assert isinstance(batch_dim, str) or batch_dim in (0, 1)

            # Spatial & channel dimensions must strictly match native training input sizes
            spatial_shape = [d.dim_value for d in dims[1:]]
            expected_spatial = spec["input_shape"][1:]
            assert spatial_shape == expected_spatial, (
                f"Model {spec['file']} spatial shape {spatial_shape} does not match native shape {expected_spatial}"
            )


class TestNumericalParityPyTorchVsONNX:
    """Verifies that ONNX Runtime produces identical numerical outputs to PyTorch eager mode."""

    @pytest.fixture(scope="class")
    def weights_dir(self):
        return REPO_ROOT / "models" / "weights"

    @pytest.fixture(scope="class")
    def onnx_engine(self):
        return ONNXInferenceEngine()

    def test_stage1_triage_numerical_parity(self, weights_dir, onnx_engine):
        torch.manual_seed(42)
        pt_model = Stage1EdgeClassifier(pretrained=False)
        pt_model.load_state_dict(torch.load(weights_dir / "stage1_mobilenetv3_india_v2.pt", map_location="cpu", weights_only=True))
        pt_model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pt_out = pt_model(dummy).numpy()

        ort_out = onnx_engine.sessions["stage1"].run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        assert max_diff < 1e-4, f"Stage 1 parity diff {max_diff} exceeded 1e-4"

    def test_stage2_structural_numerical_parity(self, weights_dir, onnx_engine):
        torch.manual_seed(42)
        pt_model = StructuralDamageHead(load_weights=False)
        pt_model.load_state_dict(torch.load(weights_dir / "stage2_structural_rescuenet_v2.pt", map_location="cpu", weights_only=True))
        pt_model.eval()

        dummy = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            pt_out = pt_model(dummy).numpy()

        ort_out = onnx_engine.sessions["structural"].run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        assert max_diff < 1e-4, f"Structural parity diff {max_diff} exceeded 1e-4"

    def test_stage2_road_numerical_parity(self, weights_dir, onnx_engine):
        torch.manual_seed(42)
        pt_model = RoadPassabilityClassifier(load_weights=False)
        pt_model.load_state_dict(torch.load(weights_dir / "stage2_road_passability_v2.pt", map_location="cpu", weights_only=True))
        pt_model.eval()

        dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            pt_out = pt_model(dummy).numpy()

        ort_out = onnx_engine.sessions["road"].run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        assert max_diff < 1e-4, f"Road parity diff {max_diff} exceeded 1e-4"

    def test_stage2_flood_unet_numerical_parity_and_mask_match(self, weights_dir, onnx_engine):
        """Per User Guidance 2: For U-Net segmentation, verify both logit tolerance (< 1e-3)
        AND exact thresholded water mask agreement (thresholded at logit 0.0 / prob 0.5).
        """
        torch.manual_seed(42)
        pt_model = FloodSegmentationUNet(load_weights=False)
        pt_model.load_state_dict(torch.load(weights_dir / "stage2_flood_unet_v2.pt", map_location="cpu", weights_only=True))
        pt_model.eval()

        dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            pt_logits = pt_model(dummy).numpy()

        ort_logits = onnx_engine.sessions["flood_unet"].run(None, {"input": dummy.numpy()})[0]

        max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
        assert max_diff < 1e-3, f"Flood U-Net logit parity diff {max_diff} exceeded 1e-3"

        # Thresholded water mask equivalence (logit > 0 corresponds to prob > 0.5)
        pt_mask = (pt_logits > 0.0).astype(np.uint8)
        ort_mask = (ort_logits > 0.0).astype(np.uint8)
        assert np.array_equal(pt_mask, ort_mask), "Thresholded water masks differed between PyTorch and ONNX"


class TestEdgePipelineAndEarlyExit:
    """Validates end-to-end edge pipeline execution and early-exit routing."""

    @pytest.fixture(scope="class")
    def onnx_engine(self):
        return ONNXInferenceEngine()

    def test_normal_scene_early_exit(self, onnx_engine):
        """Ensures normal patrol frames trigger early exit, bypassing secondary compute."""
        normal_path = REPO_ROOT / "data" / "AIDER" / "AIDER" / "normal" / "normal_image0001.jpg"
        if normal_path.exists():
            frame = cv2.cvtColor(cv2.imread(str(normal_path)), cv2.COLOR_BGR2RGB)
        else:
            frame = np.full((512, 512, 3), [80, 160, 80], dtype=np.uint8)

        res = onnx_engine.run_edge_pipeline(frame)
        assert res["disaster_class"] == "normal_scene"
        assert res["early_exit"] is True
        assert res["stage2"]["bypassed"] is True
        assert "stage2_flood_unet_ms" not in res["latency_breakdown_ms"]
        assert "stage2_structural_ms" not in res["latency_breakdown_ms"]

    def test_flood_scene_full_pipeline(self, onnx_engine):
        """Ensures flood incident frames route through secondary segmentation and passability."""
        flood_path = REPO_ROOT / "data" / "AIDER" / "AIDER" / "flooded_areas" / "flood_image0440.jpg"
        if not flood_path.exists():
            pytest.skip(f"Test flood image {flood_path} not found")

        frame = cv2.cvtColor(cv2.imread(str(flood_path)), cv2.COLOR_BGR2RGB)
        res = onnx_engine.run_edge_pipeline(frame)

        assert res["disaster_class"] == "flood"
        assert res["early_exit"] is False
        assert res["stage2"]["bypassed"] is False
        assert "water_extent_percentage" in res["stage2"]
        assert "road_passability" in res["stage2"]
        assert "stage2_flood_unet_ms" in res["latency_breakdown_ms"]
        assert "stage2_road_ms" in res["latency_breakdown_ms"]

    def test_early_exit_latency_differential(self, onnx_engine):
        """Per User Guidance 3: Verifies that early-exit normal patrol latency is measurably faster
        than full multi-stage flood pipeline execution.
        """
        normal_path = REPO_ROOT / "data" / "AIDER" / "AIDER" / "normal" / "normal_image0001.jpg"
        flood_path = REPO_ROOT / "data" / "AIDER" / "AIDER" / "flooded_areas" / "flood_image0440.jpg"

        if not normal_path.exists() or not flood_path.exists():
            pytest.skip("Test images not found")

        normal_frame = cv2.cvtColor(cv2.imread(str(normal_path)), cv2.COLOR_BGR2RGB)
        flood_frame = cv2.cvtColor(cv2.imread(str(flood_path)), cv2.COLOR_BGR2RGB)

        # Warmup multiple iterations to reach steady state
        for _ in range(3):
            onnx_engine.run_edge_pipeline(normal_frame)
            onnx_engine.run_edge_pipeline(flood_frame)

        normal_latencies = []
        flood_latencies = []
        for _ in range(10):
            normal_latencies.append(onnx_engine.run_edge_pipeline(normal_frame)["total_latency_ms"])
            flood_latencies.append(onnx_engine.run_edge_pipeline(flood_frame)["total_latency_ms"])

        mean_normal = float(np.mean(normal_latencies))
        mean_flood = float(np.mean(flood_latencies))

        # Normal patrol must be measurably faster than flood incident because it bypasses Stage 2
        assert mean_normal < mean_flood, (
            f"Early-exit failed to reduce latency: Normal mean {mean_normal:.2f} ms >= Flood mean {mean_flood:.2f} ms"
        )


class TestTemperatureCalibrationOptimization:
    """Validates TemperatureScaler.fit() and empirical ECE reduction."""

    def test_temperature_scaler_fit_reduces_ece(self):
        torch.manual_seed(42)
        np.random.seed(42)

        val_n = 250
        true_labels = torch.randint(0, 8, (val_n,))
        # Produce overconfident uncalibrated logits
        raw_logits = torch.randn(val_n, 8) * 3.5
        for i in range(val_n):
            raw_logits[i, true_labels[i]] += 2.2

        with torch.no_grad():
            pre_probs = torch.softmax(raw_logits, dim=-1).numpy()
        pre_confs = np.max(pre_probs, axis=-1)
        pre_preds = np.argmax(pre_probs, axis=-1)
        pre_ece = ReliabilityEvaluator.compute_ece(pre_confs, pre_preds, true_labels.numpy())

        scaler = TemperatureScaler(init_temperature=1.0)
        fitted_t = scaler.fit(raw_logits, true_labels)

        assert fitted_t > 1.0, f"Expected temperature > 1.0 to soften overconfident logits, got {fitted_t}"

        with torch.no_grad():
            post_logits = scaler(raw_logits)
            post_probs = torch.softmax(post_logits, dim=-1).numpy()
        post_confs = np.max(post_probs, axis=-1)
        post_preds = np.argmax(post_probs, axis=-1)
        post_ece = ReliabilityEvaluator.compute_ece(post_confs, post_preds, true_labels.numpy())

        # Assert rank preservation: temperature scaling must not alter predictions
        assert np.array_equal(pre_preds, post_preds), "Temperature scaling changed model class predictions!"

        # Assert ECE reduction
        assert post_ece["expected_calibration_error"] < pre_ece["expected_calibration_error"], (
            f"ECE did not decrease: post {post_ece['expected_calibration_error']} >= pre {pre_ece['expected_calibration_error']}"
        )


class TestHardwareTransparencyAndAuditTrail:
    """Validates that edge latency benchmarks maintain honest scoping and audit logging."""

    def test_benchmark_report_hardware_scoping(self):
        report_path = REPO_ROOT / "reports" / "edge_latency_benchmark.json"
        assert report_path.exists(), "Benchmark report missing"

        data = json.loads(report_path.read_text(encoding="utf-8"))
        hw_scope = data.get("hardware_scope", {})

        assert "Host CPU" in hw_scope.get("execution_environment", "")
        assert "NOT ATTEMPTED" in hw_scope.get("jetson_orin_nano_status", "")

        # All 4 components must meet the 50ms budget
        comp_b = data.get("component_benchmarks", {})
        for k, comp in comp_b.items():
            ort_mean = comp["onnx_runtime"]["mean_ms"]
            assert ort_mean <= 50.0, f"{comp['name']} ONNX mean latency {ort_mean} ms exceeded 50ms budget"
            assert comp["speedup_ratio"] > 1.0, f"{comp['name']} speedup ratio {comp['speedup_ratio']} <= 1.0"

"""ONNX Runtime Inference Engine for High-Performance Edge Deployments.
Loads exported ONNX graphs into onnxruntime.InferenceSession with all graph optimizations enabled.
Provides individual stage prediction APIs and an end-to-end edge triage pipeline with
early-exit optimization for normal/non-disaster scenes.
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import time
import numpy as np
import cv2
import onnxruntime as ort

from data_pipeline.schema import DisasterClass, DamageGrade, RoadPassability


class ONNXInferenceEngine:
    """Production ONNX Runtime execution engine for UAV companion computers."""

    DEFAULT_ONNX_DIR = Path(__file__).resolve().parent / "onnx"

    CLASSES = [
        DisasterClass.FLOOD,
        DisasterClass.CYCLONE_STORM,
        DisasterClass.EARTHQUAKE_COLLAPSE,
        DisasterClass.LANDSLIDE,
        DisasterClass.WILDFIRE,
        DisasterClass.INDUSTRIAL_CHEMICAL,
        DisasterClass.DROUGHT_HEATWAVE,
        DisasterClass.NORMAL_SCENE
    ]

    def __init__(self, onnx_dir: Optional[Path] = None, num_threads: int = 2):
        self.onnx_dir = Path(onnx_dir) if onnx_dir else self.DEFAULT_ONNX_DIR
        self.num_threads = num_threads

        # Configure session options for maximum performance
        self.opts = ort.SessionOptions()
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.opts.intra_op_num_threads = num_threads
        self.opts.inter_op_num_threads = 1
        self.opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.sessions: Dict[str, ort.InferenceSession] = {}
        self._init_sessions()
        self._warmup_sessions()

    def _init_sessions(self):
        providers = ["CPUExecutionProvider"]
        models_map = {
            "stage1": self.onnx_dir / "stage1_mobilenetv3_triage_v2.onnx",
            "structural": self.onnx_dir / "stage2_structural_rescuenet_v2.onnx",
            "road": self.onnx_dir / "stage2_road_passability_v2.onnx",
            "flood_unet": self.onnx_dir / "stage2_flood_unet_v2.onnx"
        }

        for key, p in models_map.items():
            if p.exists():
                self.sessions[key] = ort.InferenceSession(str(p), sess_options=self.opts, providers=providers)

    def _warmup_sessions(self):
        """Pre-warms all loaded ONNX graphs so initial runtime calls do not suffer JIT allocation latency."""
        dummy_shapes = {
            "stage1": (1, 3, 224, 224),
            "structural": (1, 3, 64, 64),
            "road": (1, 3, 128, 128),
            "flood_unet": (1, 3, 128, 128)
        }
        for key, shape in dummy_shapes.items():
            if key in self.sessions:
                dummy = np.zeros(shape, dtype=np.float32)
                self.sessions[key].run(None, {"input": dummy})

    # -------------------------------------------------------------
    # Individual Stage Predictors
    # -------------------------------------------------------------
    def predict_triage(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Runs Stage 1 MobileNetV3 triage on a 224x224 RGB aerial frame."""
        if "stage1" not in self.sessions:
            raise RuntimeError("Stage 1 ONNX model not loaded")

        t0 = time.perf_counter_ns()
        resized = cv2.resize(image_rgb, (224, 224))
        tensor = resized.astype(np.float32) / 255.0
        # ImageNet standardization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[np.newaxis, ...])

        logits = self.sessions["stage1"].run(None, {"input": tensor})[0][0]

        # Softmax & Entropy
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        top_idx = int(np.argmax(probs))
        conf = float(probs[top_idx])

        eps = 1e-12
        entropy = float(-np.sum(probs * np.log(probs + eps)))
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        d_class = self.CLASSES[top_idx]
        if conf < 0.65:
            d_class = DisasterClass.UNCERTAIN_NEEDS_REVIEW

        return {
            "disaster_class": d_class,
            "confidence": round(conf, 4),
            "entropy": round(entropy, 4),
            "probabilities": {c.value: round(float(probs[i]), 4) for i, c in enumerate(self.CLASSES)},
            "latency_ms": round(latency_ms, 3)
        }

    def assess_structural_damage(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """Grades structural damage on a 64x64 building crop."""
        if "structural" not in self.sessions:
            raise RuntimeError("Structural ONNX model not loaded")

        t0 = time.perf_counter_ns()
        resized = cv2.resize(crop_bgr, (64, 64))
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[np.newaxis, ...])

        logits = self.sessions["structural"].run(None, {"input": tensor})[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        grade_idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        return {
            "damage_grade": DamageGrade(grade_idx),
            "confidence": round(float(probs[grade_idx]), 4),
            "grade_probabilities": [round(float(p), 4) for p in probs],
            "latency_ms": round(latency_ms, 3)
        }

    def classify_road_passability(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Evaluates road passability on a 128x128 road patch."""
        if "road" not in self.sessions:
            raise RuntimeError("Road passability ONNX model not loaded")

        t0 = time.perf_counter_ns()
        resized = cv2.resize(image_rgb, (128, 128))
        tensor = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[np.newaxis, ...])

        logits = self.sessions["road"].run(None, {"input": tensor})[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        # Axiom 3 Uncertainty Gating
        if conf < 0.65:
            passability = RoadPassability.UNCERTAIN_NEEDS_REVIEW
        else:
            passability = RoadPassability.ROAD_BLOCKED if pred_idx == 1 else RoadPassability.ROAD_CLEAR

        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        return {
            "road_passability": passability,
            "confidence": round(conf, 4),
            "clear_prob": round(float(probs[0]), 4),
            "blocked_prob": round(float(probs[1]), 4),
            "latency_ms": round(latency_ms, 3)
        }

    def segment_flood_water(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Runs Flood U-Net on a 128x128 patch to extract surface water extent."""
        if "flood_unet" not in self.sessions:
            raise RuntimeError("Flood U-Net ONNX model not loaded")

        t0 = time.perf_counter_ns()
        resized = cv2.resize(image_rgb, (128, 128))
        tensor = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[np.newaxis, ...])

        mask_logits = self.sessions["flood_unet"].run(None, {"input": tensor})[0][0, 0]
        # Sigmoid activation
        probs = 1.0 / (1.0 + np.exp(-mask_logits))
        binary_mask = (probs > 0.5).astype(np.uint8)
        water_extent_pct = float(np.mean(binary_mask) * 100.0)
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        return {
            "water_extent_percentage": round(water_extent_pct, 2),
            "water_pixels": int(np.sum(binary_mask)),
            "total_pixels": int(binary_mask.size),
            "binary_mask": binary_mask,
            "latency_ms": round(latency_ms, 3)
        }

    # -------------------------------------------------------------
    # End-to-End Edge Pipeline (with Early Exit)
    # -------------------------------------------------------------
    def run_edge_pipeline(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """Executes full on-drone edge perception pipeline.
        Implements early exit: normal scenes bypass Stage 2 completely.
        """
        pipeline_t0 = time.perf_counter_ns()
        stage_latencies = {}

        # Step 1: Stage 1 Triage
        triage = self.predict_triage(frame_rgb)
        d_class = triage["disaster_class"]
        stage_latencies["stage1_triage_ms"] = triage["latency_ms"]

        # Step 2: Routing logic
        early_exit = False
        stage2_results = {}

        if d_class == DisasterClass.NORMAL_SCENE:
            # Early Exit: zero secondary compute needed
            early_exit = True
            stage2_results = {
                "bypassed": True,
                "reason": "Normal scene detected: secondary severity analysis bypassed."
            }

        elif d_class in (DisasterClass.FLOOD, DisasterClass.CYCLONE_STORM):
            # Flood path: segmentation + corridor passability
            flood_res = self.segment_flood_water(frame_rgb)
            road_res = self.classify_road_passability(frame_rgb)
            stage_latencies["stage2_flood_unet_ms"] = flood_res["latency_ms"]
            stage_latencies["stage2_road_ms"] = road_res["latency_ms"]
            stage2_results = {
                "bypassed": False,
                "water_extent_percentage": flood_res["water_extent_percentage"],
                "road_passability": road_res["road_passability"].value,
                "road_confidence": road_res["confidence"]
            }

        elif d_class in (DisasterClass.EARTHQUAKE_COLLAPSE, DisasterClass.LANDSLIDE):
            # Structural path: building damage grading on center crop
            h, w = frame_rgb.shape[:2]
            crop = frame_rgb[h//4:3*h//4, w//4:3*w//4]
            struct_res = self.assess_structural_damage(crop)
            stage_latencies["stage2_structural_ms"] = struct_res["latency_ms"]
            stage2_results = {
                "bypassed": False,
                "damage_grade": struct_res["damage_grade"].value,
                "structural_confidence": struct_res["confidence"]
            }

        else:
            stage2_results = {
                "bypassed": True,
                "reason": f"Disaster type {d_class.value} routed to specialized handling."
            }

        total_latency_ms = (time.perf_counter_ns() - pipeline_t0) / 1e6

        return {
            "disaster_class": d_class.value,
            "confidence": triage["confidence"],
            "early_exit": early_exit,
            "stage1": triage,
            "stage2": stage2_results,
            "latency_breakdown_ms": stage_latencies,
            "total_latency_ms": round(total_latency_ms, 3),
            "meets_50ms_budget": total_latency_ms <= 50.0
        }

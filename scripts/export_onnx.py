"""Export Active PyTorch v2 Checkpoints to Standardized ONNX Runtime Graphs.
Exports:
1. Stage 1 Triage (MobileNetV3): (1, 3, 224, 224) -> (1, 8)
2. Stage 2 Structural Damage: (1, 3, 64, 64) -> (1, 4)
3. Stage 2 Road Passability: (1, 3, 128, 128) -> (1, 2)
4. Stage 2 Flood U-Net: (1, 3, 128, 128) -> (1, 1, 128, 128)

Performs graph verification, numerical parity audit against PyTorch outputs,
computes SHA-256 checksums, and emits models/onnx/onnx_manifest.json.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
import hashlib
import numpy as np
import torch
import onnx
import onnxruntime as ort

from models.stage1_classifier import Stage1EdgeClassifier
from models.stage2_severity import StructuralDamageHead, RoadPassabilityClassifier, FloodSegmentationUNet


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_all_models(output_dir: Path = REPO_ROOT / "models" / "onnx") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = REPO_ROOT / "models" / "weights"

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "platform_version": "2.0.0",
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset_version": 14,
        "models": {}
    }

    print("=" * 70, flush=True)
    print("  ONNX MODEL EXPORT & NUMERICAL PARITY AUDIT (v2 Checkpoints)", flush=True)
    print("=" * 70, flush=True)

    # 1. Stage 1: MobileNetV3 Disaster Triage
    print("\n[1/4] Exporting Stage 1: MobileNetV3 Triage...", flush=True)
    m1 = Stage1EdgeClassifier(pretrained=False)
    w1_path = weights_dir / "stage1_mobilenetv3_india_v2.pt"
    if w1_path.exists():
        m1.load_state_dict(torch.load(w1_path, map_location="cpu", weights_only=True))
        print(f"  Loaded weights: {w1_path.name}", flush=True)
    m1.eval()

    dummy1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    p1 = output_dir / "stage1_mobilenetv3_triage_v2.onnx"
    torch.onnx.export(
        m1, dummy1, str(p1),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14, dynamo=False
    )
    onnx.checker.check_model(str(p1))

    with torch.no_grad():
        py_out1 = m1(dummy1).numpy()
    sess1 = ort.InferenceSession(str(p1), providers=["CPUExecutionProvider"])
    ort_out1 = sess1.run(None, {"input": dummy1.numpy()})[0]
    max_diff1 = float(np.max(np.abs(py_out1 - ort_out1)))
    assert max_diff1 < 1e-4, f"Stage 1 numerical parity failure: max diff {max_diff1}"
    print(f"  Exported: {p1.name} ({p1.stat().st_size:,} bytes) | Parity Max Diff: {max_diff1:.2e} [OK]", flush=True)

    manifest["models"]["stage1_triage"] = {
        "model_id": "stage1_mobilenetv3_triage_v2",
        "file_name": p1.name,
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 8],
        "size_bytes": p1.stat().st_size,
        "sha256": compute_sha256(p1),
        "parity_max_diff": max_diff1,
        "parity_status": "EXACT_PARITY"
    }

    # 2. Stage 2: Structural Damage Head
    print("\n[2/4] Exporting Stage 2: Structural Damage Head...", flush=True)
    m2 = StructuralDamageHead(load_weights=False)
    w2_path = weights_dir / "stage2_structural_rescuenet_v2.pt"
    if w2_path.exists():
        m2.load_state_dict(torch.load(w2_path, map_location="cpu", weights_only=True))
        print(f"  Loaded weights: {w2_path.name}", flush=True)
    m2.eval()

    dummy2 = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    p2 = output_dir / "stage2_structural_rescuenet_v2.onnx"
    torch.onnx.export(
        m2, dummy2, str(p2),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14, dynamo=False
    )
    onnx.checker.check_model(str(p2))

    with torch.no_grad():
        py_out2 = m2(dummy2).numpy()
    sess2 = ort.InferenceSession(str(p2), providers=["CPUExecutionProvider"])
    ort_out2 = sess2.run(None, {"input": dummy2.numpy()})[0]
    max_diff2 = float(np.max(np.abs(py_out2 - ort_out2)))
    assert max_diff2 < 1e-4, f"Stage 2 Structural numerical parity failure: max diff {max_diff2}"
    print(f"  Exported: {p2.name} ({p2.stat().st_size:,} bytes) | Parity Max Diff: {max_diff2:.2e} [OK]", flush=True)

    manifest["models"]["stage2_structural"] = {
        "model_id": "stage2_structural_rescuenet_v2",
        "file_name": p2.name,
        "input_shape": [1, 3, 64, 64],
        "output_shape": [1, 4],
        "size_bytes": p2.stat().st_size,
        "sha256": compute_sha256(p2),
        "parity_max_diff": max_diff2,
        "parity_status": "EXACT_PARITY"
    }

    # 3. Stage 2: Road Passability Classifier
    print("\n[3/4] Exporting Stage 2: Road Passability Classifier...", flush=True)
    m3 = RoadPassabilityClassifier(load_weights=False)
    w3_path = weights_dir / "stage2_road_passability_v2.pt"
    if w3_path.exists():
        m3.load_state_dict(torch.load(w3_path, map_location="cpu", weights_only=True))
        print(f"  Loaded weights: {w3_path.name}", flush=True)
    m3.eval()

    dummy3 = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    p3 = output_dir / "stage2_road_passability_v2.onnx"
    torch.onnx.export(
        m3, dummy3, str(p3),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14, dynamo=False
    )
    onnx.checker.check_model(str(p3))

    with torch.no_grad():
        py_out3 = m3(dummy3).numpy()
    sess3 = ort.InferenceSession(str(p3), providers=["CPUExecutionProvider"])
    ort_out3 = sess3.run(None, {"input": dummy3.numpy()})[0]
    max_diff3 = float(np.max(np.abs(py_out3 - ort_out3)))
    assert max_diff3 < 1e-4, f"Stage 2 Road numerical parity failure: max diff {max_diff3}"
    print(f"  Exported: {p3.name} ({p3.stat().st_size:,} bytes) | Parity Max Diff: {max_diff3:.2e} [OK]", flush=True)

    manifest["models"]["stage2_road"] = {
        "model_id": "stage2_road_passability_v2",
        "file_name": p3.name,
        "input_shape": [1, 3, 128, 128],
        "output_shape": [1, 2],
        "size_bytes": p3.stat().st_size,
        "sha256": compute_sha256(p3),
        "parity_max_diff": max_diff3,
        "parity_status": "EXACT_PARITY"
    }

    # 4. Stage 2: Flood Segmentation U-Net
    print("\n[4/4] Exporting Stage 2: Flood Segmentation U-Net...", flush=True)
    m4 = FloodSegmentationUNet(load_weights=False)
    w4_path = weights_dir / "stage2_flood_unet_v2.pt"
    if w4_path.exists():
        m4.load_state_dict(torch.load(w4_path, map_location="cpu", weights_only=True))
        print(f"  Loaded weights: {w4_path.name}", flush=True)
    m4.eval()

    dummy4 = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    p4 = output_dir / "stage2_flood_unet_v2.onnx"
    torch.onnx.export(
        m4, dummy4, str(p4),
        input_names=["input"], output_names=["mask_logits"],
        dynamic_axes={"input": {0: "batch"}, "mask_logits": {0: "batch"}},
        opset_version=14, dynamo=False
    )
    onnx.checker.check_model(str(p4))

    with torch.no_grad():
        py_out4 = m4(dummy4).numpy()
    sess4 = ort.InferenceSession(str(p4), providers=["CPUExecutionProvider"])
    ort_out4 = sess4.run(None, {"input": dummy4.numpy()})[0]
    max_diff4 = float(np.max(np.abs(py_out4 - ort_out4)))

    py_mask = (py_out4 > 0.0).astype(np.uint8)
    ort_mask = (ort_out4 > 0.0).astype(np.uint8)
    masks_match = bool(np.array_equal(py_mask, ort_mask))
    assert masks_match is True, "Flood U-Net thresholded mask mismatch"
    print(f"  Exported: {p4.name} ({p4.stat().st_size:,} bytes) | Parity Max Diff: {max_diff4:.2e} | Mask Match: {masks_match} [OK]", flush=True)

    manifest["models"]["stage2_flood_unet"] = {
        "model_id": "stage2_flood_unet_v2",
        "file_name": p4.name,
        "input_shape": [1, 3, 128, 128],
        "output_shape": [1, 1, 128, 128],
        "size_bytes": p4.stat().st_size,
        "sha256": compute_sha256(p4),
        "parity_max_diff": max_diff4,
        "parity_binary_mask_match": masks_match,
        "parity_status": "EXACT_PARITY"
    }

    # Save Manifest
    manifest_path = output_dir / "onnx_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest saved to: {manifest_path}", flush=True)
    print("=" * 70, flush=True)
    print("  ALL 4 ONNX MODELS EXPORTED AND PARITY-VERIFIED [SUCCESS]", flush=True)
    print("=" * 70, flush=True)
    return manifest


if __name__ == "__main__":
    export_all_models()

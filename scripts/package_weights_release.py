"""Package and Verify Model Weights for Release.
Verifies SHA-256 checksums of all weights in models/weights/ against
config/model_registry.json, generates an immutable release manifest,
and packages them into a portable release archive.
"""

import json
import hashlib
import zipfile
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "models" / "weights"
REGISTRY_PATH = ROOT / "config" / "model_registry.json"
OUTPUT_ZIP = ROOT / "disaster_response_v2_weights.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("DISASTER RESPONSE AI — WEIGHTS RELEASE PACKAGING & VERIFICATION")
    print("=" * 70)

    if not REGISTRY_PATH.exists():
        print(f"[ERROR] Registry not found at {REGISTRY_PATH}")
        sys.exit(1)

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    models = registry.get("models", {})
    manifest = {
        "version": "2.0.0",
        "total_models": len(models),
        "active_models": registry.get("active_models", {}),
        "files": []
    }

    print(f"\n[1] Verifying {len(models)} registered models against local files...")
    verified_count = 0
    missing_count = 0
    all_weights_paths = set()

    for mid, meta in models.items():
        w_path_str = meta.get("weights_path")
        if not w_path_str or w_path_str == "none":
            continue

        p = Path(w_path_str)
        if not p.is_absolute():
            p = ROOT / p

        expected_hash = meta.get("sha256_checksum")
        if not p.exists():
            print(f"  [MISSING] {mid}: {p.name}")
            missing_count += 1
            continue

        all_weights_paths.add(p)
        actual_hash = sha256_file(p)
        status = "OK" if actual_hash.lower() == expected_hash.lower() else "MISMATCH"
        print(f"  [{status}] {mid} -> {p.name} ({p.stat().st_size:,} bytes)")
        if status != "OK":
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {actual_hash}")

        manifest["files"].append({
            "model_id": mid,
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": actual_hash,
            "is_active": meta.get("is_active", False),
            "stage": meta.get("stage"),
            "architecture": meta.get("architecture")
        })
        verified_count += 1

    # Also capture any other .pt files in models/weights
    for wf in sorted(WEIGHTS_DIR.glob("*.pt")):
        if wf not in all_weights_paths:
            h = sha256_file(wf)
            manifest["files"].append({
                "model_id": wf.stem,
                "filename": wf.name,
                "size_bytes": wf.stat().st_size,
                "sha256": h,
                "is_active": False,
                "stage": "unregistered_or_legacy",
                "architecture": "PyTorch Checkpoint"
            })
            print(f"  [EXTRA] {wf.name} ({wf.stat().st_size:,} bytes)")

    # Write manifest
    manifest_path = ROOT / "models" / "weights_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[2] Manifest saved: {manifest_path}")

    # Build Zip Archive
    print(f"\n[3] Creating release archive: {OUTPUT_ZIP.name}...")
    total_uncompressed = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add manifest
        zf.write(manifest_path, arcname="weights_manifest.json")
        # Add registry
        zf.write(REGISTRY_PATH, arcname="model_registry.json")
        # Add all weights
        for wf in sorted(WEIGHTS_DIR.glob("*.pt")):
            zf.write(wf, arcname=f"weights/{wf.name}")
            total_uncompressed += wf.stat().st_size

    zip_size = OUTPUT_ZIP.stat().st_size
    ratio = (1.0 - (zip_size / total_uncompressed)) * 100 if total_uncompressed > 0 else 0.0

    print(f"  Archive created: {OUTPUT_ZIP.stat().st_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"  Uncompressed weights: {total_uncompressed / (1024*1024):.2f} MB (Compression: {ratio:.1f}%)")
    print(f"  SHA-256: {sha256_file(OUTPUT_ZIP)}")
    print("\n[OK] Release bundle ready for GitHub Releases upload.")


if __name__ == "__main__":
    main()

"""Cherry-pick merge: copy artifact weights + update registry.

Steps 3-4 of the merge plan:
  3. Copy stage2_structural_rescuenet_v2.pt and stage2_road_passability_v2.pt
     from disaster_response_v2_artifacts/ into models/weights/.
  4. Add both v2 entries to the registry, set active_models pointers,
     and preserve the repo's existing (better) Stage 1 and Flood entries.
"""

import json
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent  # Disaster Response/
ARTIFACT_DIR = ROOT / "disaster_response_v2_artifacts"
WEIGHTS_DIR = ROOT / "models" / "weights"
REGISTRY_PATH = ROOT / "config" / "model_registry.json"
ARTIFACT_REGISTRY_PATH = ARTIFACT_DIR / "config" / "model_registry.json"

# ── Files to cherry-pick from artifact ──────────────────────────
CHERRYPICK = [
    "stage2_structural_rescuenet_v2.pt",
    "stage2_road_passability_v2.pt",
]

# ── Models to import into the registry from artifact ────────────
IMPORT_MODEL_IDS = [
    "stage2_structural_rescuenet_v2",
    "stage2_road_passability_v2",
]

# ── Active model pointer updates ────────────────────────────────
ACTIVE_UPDATES = {
    "stage2_severity": "stage2_structural_rescuenet_v2",
    "stage2_road_passability": "stage2_road_passability_v2",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 65)
    print("CHERRY-PICK MERGE: Artifact -> Repo")
    print("=" * 65)

    # ── Step 3: Copy weight files ────────────────────────────────
    print("\n[STEP 3] Copying weight files...")
    for fname in CHERRYPICK:
        src = ARTIFACT_DIR / "models" / "weights" / fname
        dst = WEIGHTS_DIR / fname
        if not src.exists():
            print(f"  [ERROR] Source not found: {src}")
            sys.exit(1)
        if dst.exists():
            print(f"  [WARN] {fname} already exists in repo, overwriting")
        shutil.copy2(src, dst)
        # Verify copy integrity
        src_hash = sha256_file(src)
        dst_hash = sha256_file(dst)
        if src_hash != dst_hash:
            print(f"  [ERROR] Copy verification failed for {fname}!")
            sys.exit(1)
        print(f"  [OK] {fname} ({dst.stat().st_size:,} bytes, SHA256: {dst_hash[:16]}...)")

    # ── Step 4: Update registry ──────────────────────────────────
    print("\n[STEP 4] Updating model registry...")

    # Load both registries
    with open(REGISTRY_PATH, "r") as f:
        repo_reg = json.load(f)
    with open(ARTIFACT_REGISTRY_PATH, "r") as f:
        art_reg = json.load(f)

    # Import model entries from artifact registry
    for mid in IMPORT_MODEL_IDS:
        if mid not in art_reg["models"]:
            print(f"  [ERROR] {mid} not found in artifact registry!")
            sys.exit(1)

        entry = art_reg["models"][mid]

        # If model_id already exists in repo, preserve it but update
        if mid in repo_reg["models"]:
            print(f"  [UPDATE] {mid} (already existed, overwriting with artifact entry)")
        else:
            print(f"  [ADD] {mid}")

        repo_reg["models"][mid] = entry

    # Deactivate previous models for stages we're updating
    for stage, new_active_id in ACTIVE_UPDATES.items():
        for mid, mdata in repo_reg["models"].items():
            if mdata.get("stage") == stage and mid != new_active_id:
                mdata["is_active"] = False

    # Set active model pointers
    for stage, model_id in ACTIVE_UPDATES.items():
        repo_reg["active_models"][stage] = model_id
        repo_reg["models"][model_id]["is_active"] = True
        print(f"  [ACTIVE] {stage} -> {model_id}")

    # Save updated registry
    with open(REGISTRY_PATH, "w") as f:
        json.dump(repo_reg, f, indent=2)
    print(f"\n  [OK] Registry saved to {REGISTRY_PATH}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MERGE SUMMARY")
    print("=" * 65)

    with open(REGISTRY_PATH, "r") as f:
        final = json.load(f)

    print("Active models:")
    for stage, mid in final["active_models"].items():
        meta = final["models"][mid]
        print(f"  {stage}: {mid} (v{meta['version']}, {meta['status']})")

    print("\nWeight files in models/weights/:")
    for wf in sorted(WEIGHTS_DIR.glob("*.pt")):
        print(f"  {wf.name} ({wf.stat().st_size:,} bytes)")

    print("\n[DONE] Cherry-pick merge complete.")
    print("Next: run 'python -m pytest tests/ -v' to validate.")


if __name__ == "__main__":
    main()

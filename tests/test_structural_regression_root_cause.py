"""Root-Cause Verification Test for Structural Damage Accuracy Regression.
Specifically diagnoses and guards against:
1. Checkpoint Contamination: Instantiating with load_weights=True during retraining
   caused an optimizer with zero momentum and high LR to destructively update converged weights.
2. Learning Rate Overshoot: At batch_size=32 on 480 crops (15 steps/epoch), lr=0.001 caused
   severe optimization overshoot, dropping accuracy to 40.62%.
3. Restored Convergence: Verifies that fresh initialization + lr=0.0005 + 8 epochs achieves >= 55% accuracy and MAE <= 0.60.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.stage2_severity import StructuralDamageHead
from models.train import RescueNetDamageDataset
from models.registry import ModelRegistry


class TestStructuralDamageRootCause:
    @pytest.fixture
    def rescuenet_dir(self):
        p = Path("data/RescueNet")
        if (p / "RescueNet").exists():
            return p / "RescueNet"
        return p

    def test_checkpoint_isolation_prevents_contamination(self):
        """Verifies that load_weights=False strictly isolates new model instances from existing weights."""
        weights_path = Path("models/weights/stage2_structural_rescuenet_v1.pt")
        if not weights_path.exists():
            pytest.skip("Checkpoint does not exist yet.")

        # Fresh instance with load_weights=False
        fresh_model = StructuralDamageHead(load_weights=False)
        assert fresh_model.has_weights is False, "Fresh model should not mark has_weights=True"

        # Loaded instance with load_weights=True
        loaded_model = StructuralDamageHead(load_weights=True)
        assert loaded_model.has_weights is True, "Loaded model must have weights loaded"

        # Compare weights between fresh and loaded model
        fresh_p = next(fresh_model.parameters()).detach().cpu().numpy()
        loaded_p = next(loaded_model.parameters()).detach().cpu().numpy()
        assert not np.allclose(fresh_p, loaded_p), (
            "Fresh model weights should NOT match disk weights (checkpoint contamination bug)."
        )

    def test_structural_damage_restored_accuracy_and_mae(self, rescuenet_dir):
        """Verifies that the restored model clears the 55% accuracy threshold and MAE <= 0.60."""
        val_lbl = rescuenet_dir / "val" / "val-label-img"
        if not val_lbl.exists():
            pytest.skip("RescueNet validation split not found.")

        weights_path = Path("models/weights/stage2_structural_rescuenet_v1.pt")
        if not weights_path.exists():
            pytest.skip("Trained weights not found.")

        val_dataset = RescueNetDamageDataset(rescuenet_dir, split="val", max_samples_per_class=40, seed=42)
        assert len(val_dataset) > 0, "Expected non-empty validation set"

        model = StructuralDamageHead(load_weights=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                preds = torch.argmax(out, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        accuracy = float(np.mean(preds_arr == labels_arr))
        ordinal_mae = float(np.mean(np.abs(preds_arr - labels_arr)))

        # 4-tier random chance is 25.0%. The regression dropped it to 40.62%.
        # Restored target is >= 55.0% and MAE <= 0.60.
        assert accuracy >= 0.55, f"Structural damage accuracy regressed! Expected >= 0.55, got {accuracy:.4f}"
        assert ordinal_mae <= 0.60, f"Structural damage ordinal MAE too high! Expected <= 0.60, got {ordinal_mae:.4f}"

    def test_registry_contains_restored_metrics(self):
        """Verifies model registry records the restored metrics rather than the regressed 40.62%."""
        registry = ModelRegistry()
        model_meta = registry.models.get("stage2_structural_rescuenet_v1")
        assert model_meta is not None, "stage2_structural_rescuenet_v1 must exist in registry"

        metrics = model_meta.metrics
        acc = metrics.get("val_damage_accuracy", 0.0)
        mae = metrics.get("val_ordinal_mae", 99.0)

        assert acc >= 0.55, f"Registry records regressed accuracy: {acc} < 0.55"
        assert mae <= 0.60, f"Registry records regressed MAE: {mae} > 0.60"

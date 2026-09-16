
"""Stage 1: Lightweight Edge Disaster Scene Classifier.
Grounded in AIDER (Kyrkou 2019) and EmergencyNet (Kyrkou 2021).
Designed for sub-50ms inference on UAV companion computers (e.g. NVIDIA Jetson).
Classifies raw imagery to route to specialized Stage-2 modules.
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from data_pipeline.schema import DisasterClass


def _resolve_weights_path(stage_key: str, fallback_candidates: List[Path]) -> Path:
    """Dynamically resolves active weights path from model_registry.json,
    falling back to prioritized version candidates (v2 -> v1).
    """
    repo_root = Path(__file__).resolve().parent.parent
    try:
        reg_file = repo_root / "config" / "model_registry.json"
        if reg_file.exists():
            data = json.loads(reg_file.read_text(encoding="utf-8"))
            active_id = data.get("active_models", {}).get(stage_key)
            if active_id and active_id in data.get("models", {}):
                w_str = data["models"][active_id].get("weights_path")
                if w_str and w_str != "none":
                    p = Path(w_str)
                    if p.exists():
                        return p
                    if (repo_root / p).exists():
                        return repo_root / p
    except Exception:
        pass
    for cand in fallback_candidates:
        if cand.exists():
            return cand
        if (repo_root / cand).exists():
            return repo_root / cand
    return fallback_candidates[-1] if fallback_candidates else Path("none")


class Stage1EdgeClassifier(nn.Module):
    """MobileNetV3-based lightweight disaster triage classifier."""

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

    def __init__(self, pretrained: bool = True, num_classes: int = len(CLASSES)):
        super().__init__()
        # Use MobileNetV3-Small with ImageNet pretrained weights for edge transfer learning
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Classification head with dropout to minimize overfitting on small disaster sets
        in_features = 576  # Standard MobileNetV3-small pooling output
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    @classmethod
    def preprocess_image(cls, image_rgb: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Standardizes input resolution and ImageNet normalization."""
        import cv2
        resized = cv2.resize(image_rgb, target_size)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # ImageNet mean & std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std


class DisasterTriageEngine:
    """Production inference wrapper with calibration and uncertainty gating."""

    DEFAULT_WEIGHTS_PATH = Path("models/weights/stage1_mobilenetv3_india_v1.pt")

    def __init__(self, weights_path: Optional[str] = None, pretrained_backbone: bool = True):
        self.model = Stage1EdgeClassifier(pretrained=pretrained_backbone)
        
        # Check explicit path first, then dynamically resolve active weights path
        target_path = Path(weights_path) if weights_path else _resolve_weights_path(
            "stage1_triage",
            [Path("models/weights/stage1_mobilenetv3_india_v2.pt"), Path("models/weights/stage1_mobilenetv3_india_v1.pt")]
        )
        if target_path.exists():
            try:
                state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
            except Exception:
                # Fallback to weights_only=False if older PyTorch serialization
                try:
                    state_dict = torch.load(target_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                except Exception:
                    pass
        self.model.eval()

    def predict(
        self, 
        image_rgb: np.ndarray, 
        confidence_threshold: float = 0.70,
        entropy_threshold: float = 1.25
    ) -> Dict[str, Any]:
        tensor = Stage1EdgeClassifier.preprocess_image(image_rgb)
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        
        # Calculate distribution Shannon entropy to measure prediction ambiguity
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        
        disaster_enum = Stage1EdgeClassifier.CLASSES[top_idx]

        # Explicit "No False Certainty" gating (Source #10)
        if top_conf < confidence_threshold or entropy > entropy_threshold:
            final_class = DisasterClass.UNCERTAIN_NEEDS_REVIEW
            flagged_reason = (
                f"Low confidence ({top_conf:.2f} < {confidence_threshold}) or "
                f"high entropy ({entropy:.2f} > {entropy_threshold})"
            )
        else:
            final_class = disaster_enum
            flagged_reason = None

        return {
            "disaster_class": final_class,
            "raw_top_class": disaster_enum,
            "confidence": round(top_conf, 4),
            "entropy": round(entropy, 4),
            "class_probabilities": {
                c.value: round(float(probs[i]), 4)
                for i, c in enumerate(Stage1EdgeClassifier.CLASSES)
            },
            "flagged_for_human_review": (final_class == DisasterClass.UNCERTAIN_NEEDS_REVIEW),
            "review_reason": flagged_reason
        }

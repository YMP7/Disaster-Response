"""Calibration and Reliability Evaluation Module.
Implements Temperature Scaling, Expected Calibration Error (ECE),
and explicit uncertainty gating to prevent overconfident erroneous predictions
as warned by sensor-disagreement literature (Source #10).
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Calibrates model logits via post-hoc Platt / Temperature Scaling."""

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=0.01)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01
    ) -> float:
        """Fits temperature T on validation logits via NLL loss minimization (Guo et al. 2017)."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        try:
            optimizer.step(eval_loss)
        except Exception:
            # Fallback to Adam if LBFGS encounters singular curvature
            adam_opt = torch.optim.Adam([self.temperature], lr=lr)
            for _ in range(max_iter):
                adam_opt.zero_grad()
                scaled = self.forward(logits)
                loss = nll_criterion(scaled, labels)
                loss.backward()
                adam_opt.step()

        return float(self.temperature.item())


class ReliabilityEvaluator:
    """Computes calibration error and audits model honesty."""

    @staticmethod
    def compute_ece(
        confidences: np.ndarray, 
        predictions: np.ndarray, 
        labels: np.ndarray, 
        num_bins: int = 10
    ) -> Dict[str, float]:
        """Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0
        accuracies = predictions == labels

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += diff * prop_in_bin
                mce = max(mce, diff)

        return {
            "expected_calibration_error": round(float(ece), 4),
            "max_calibration_error": round(float(mce), 4),
            "is_well_calibrated": bool(ece < 0.10)
        }

    @staticmethod
    def audit_prediction_honesty(
        probs: np.ndarray, 
        confidence_threshold: float = 0.70,
        entropy_threshold: float = 1.25
    ) -> Tuple[bool, str]:
        """Evaluates whether prediction should be certified or rejected to Human Review."""
        max_prob = float(np.max(probs))
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

        if max_prob < confidence_threshold:
            return False, f"Confidence {max_prob:.2f} below safe threshold {confidence_threshold}"
        if entropy > entropy_threshold:
            return False, f"Distribution entropy {entropy:.2f} above threshold {entropy_threshold}"

        return True, "Prediction certified honest & confident"

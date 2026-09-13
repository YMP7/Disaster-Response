"""Anomaly, Fraud, and Discrepancy Detection Engine.
Compares imagery-derived damage grades against self-reported municipal/public loss claims.
Flags excessive fund requests or severe spatial mismatches for administrative audit.
"""

from typing import Dict, Any, List
from pydantic import BaseModel


class ClaimAuditRecord(BaseModel):
    claim_id: str
    zone_id: str
    claimed_destroyed_houses: int
    claimed_loss_inr: float
    satellite_verified_destroyed_houses: int
    satellite_verified_fund_inr: float
    discrepancy_ratio: float
    is_anomalous: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    audit_recommendation: str


class DamageAnomalyDetector:
    """Detects inflation or divergence between remote sensing imagery and reported claims."""

    def __init__(self, tolerance_ratio: float = 1.40):
        self.tolerance_ratio = tolerance_ratio  # Claims > 40% above imagery triggers flag

    def audit_claim(
        self,
        claim_id: str,
        zone_id: str,
        claimed_houses: int,
        claimed_amount_inr: float,
        imagery_verified_houses: int,
        imagery_verified_amount_inr: float
    ) -> ClaimAuditRecord:
        safe_img_amount = max(1.0, imagery_verified_amount_inr)
        discrepancy = claimed_amount_inr / safe_img_amount

        is_anomalous = discrepancy > self.tolerance_ratio
        
        if discrepancy > 2.5:
            risk = "CRITICAL"
            rec = "HOLD DISBURSEMENT. Severe mismatch: claimed losses are over 250% of aerial observations. Mandate physical re-survey by State Vigilance Commission."
        elif discrepancy > self.tolerance_ratio:
            risk = "HIGH"
            rec = "FLAG FOR RECONCILIATION. Claim exceeds imagery-derived building count. Cross-verify against GIS property tax register."
        elif discrepancy < 0.6:
            risk = "MEDIUM"
            rec = "POTENTIAL UNDER-REPORTING. Imagery shows higher structural damage than recorded claims. Dispatch ground enumerators to ensure vulnerable households are included."
        else:
            risk = "LOW"
            rec = "VERIFIED. Claim aligns with remote-sensing building footprint analysis within statistical error margins."

        return ClaimAuditRecord(
            claim_id=claim_id,
            zone_id=zone_id,
            claimed_destroyed_houses=claimed_houses,
            claimed_loss_inr=claimed_amount_inr,
            satellite_verified_destroyed_houses=imagery_verified_houses,
            satellite_verified_fund_inr=imagery_verified_amount_inr,
            discrepancy_ratio=round(discrepancy, 2),
            is_anomalous=is_anomalous,
            risk_level=risk,
            audit_recommendation=rec
        )

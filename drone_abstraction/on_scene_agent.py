"""On-Scene Drone Decision Agent with Constrained Action Set.
Analyzes live video stream frames during flight, refines severity as altitude changes,
and selects exclusively from a human-approved action set.
Enforces the mandatory Human-in-the-Loop gate for any kinetic or irreversible recommendation.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np
from pydantic import BaseModel
from data_pipeline.schema import DisasterClass, DamageGrade, TelemetryFrame
from models.stage1_classifier import DisasterTriageEngine
from models.stage2_severity import FloodSeverityHead, StructuralDamageHead


class DroneActionType(str, Enum):
    CONTINUE_SURVEY = "continue_survey"
    WIDEN_SEARCH_GRID = "widen_search_grid"
    MARK_GPS_HOTSPOT = "mark_gps_hotspot"
    REQUEST_SUPPLY_DROP = "request_supply_drop"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    RETURN_TO_HOME = "return_to_home"


class DroneActionDecision(BaseModel):
    decision_id: str
    action: DroneActionType
    confidence: float
    justification: str
    requires_human_approval: bool
    hotspot_coordinates: Optional[Dict[str, float]] = None
    telemetry_snapshot: Optional[Dict[str, Any]] = None


class OnSceneDroneAgent:
    """Embedded or edge intelligence operating alongside the flight controller."""

    def __init__(self):
        self.triage_engine = DisasterTriageEngine()
        self.flood_head = FloodSeverityHead()
        self.hotspots: List[Dict[str, Any]] = []

    def evaluate_live_frame(
        self,
        frame_rgb: np.ndarray,
        telemetry: TelemetryFrame,
        flight_status: Dict[str, Any]
    ) -> DroneActionDecision:
        """Processes on-scene frame, refines classification, and chooses constrained action."""
        # 1. Check hardware failsafes first
        if flight_status.get("in_failsafe", False) or flight_status.get("battery_pct", 100) < 25.0:
            return DroneActionDecision(
                decision_id=f"act_{telemetry.frame_id}",
                action=DroneActionType.RETURN_TO_HOME,
                confidence=1.0,
                justification="Battery below return threshold or failsafe active. Initiating RTH.",
                requires_human_approval=False,
                telemetry_snapshot=telemetry.model_dump()
            )

        # 2. Re-classify at current altitude / position
        triage = self.triage_engine.predict(frame_rgb)
        disaster_class = triage["disaster_class"]

        # If model is uncertain or multi-sensor labels disagree -> Request Human Review
        if disaster_class == DisasterClass.UNCERTAIN_NEEDS_REVIEW or triage["confidence"] < 0.70:
            return DroneActionDecision(
                decision_id=f"act_{telemetry.frame_id}",
                action=DroneActionType.REQUEST_HUMAN_REVIEW,
                confidence=triage["confidence"],
                justification=f"Ambiguous visual signature or low confidence ({triage['confidence']:.2f}). Escalate to operator.",
                requires_human_approval=True,
                telemetry_snapshot=telemetry.model_dump()
            )

        # 3. Stage 2 Severity Evaluation
        if disaster_class == DisasterClass.FLOOD:
            sev = self.flood_head.analyze(frame_rgb)
            water_pct = sev["water_extent_percentage"]
            flooded_bld = sev["buildings_flooded"]

            if flooded_bld >= 2 and water_pct > 40.0:
                hotspot = {
                    "lat": telemetry.gps.latitude,
                    "lon": telemetry.gps.longitude,
                    "alt_agl": telemetry.altitude_agl_m,
                    "water_extent_pct": water_pct,
                    "submerged_structures": flooded_bld
                }
                self.hotspots.append(hotspot)

                # Irreversible recommendation: requires Human Command Center confirmation
                return DroneActionDecision(
                    decision_id=f"act_{telemetry.frame_id}",
                    action=DroneActionType.REQUEST_SUPPLY_DROP,
                    confidence=0.92,
                    justification=f"Confirmed cluster of {flooded_bld} submerged residential units. Critical relief drop proposed.",
                    requires_human_approval=True,
                    hotspot_coordinates={"lat": telemetry.gps.latitude, "lon": telemetry.gps.longitude},
                    telemetry_snapshot=telemetry.model_dump()
                )

        # Standard operating procedure: continue surveying current mission grid
        return DroneActionDecision(
            decision_id=f"act_{telemetry.frame_id}",
            action=DroneActionType.CONTINUE_SURVEY,
            confidence=triage["confidence"],
            justification="Grid reconnaissance nominal. No immediate life-safety anomaly detected.",
            requires_human_approval=False,
            telemetry_snapshot=telemetry.model_dump()
        )

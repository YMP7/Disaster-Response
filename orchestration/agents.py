"""DisasTeller Multi-Agent Orchestration Architecture.
Grounded in DisasTeller (2024).
Decomposes response into specialized cooperating micro-agents:
- DetectionAgent: Ingestion & Stage-1 edge triage
- AssessmentAgent: Stage-2 severity, segmentation & multimodal SAR fusion
- ResourceFundsAgent: SDRF relief norms & anomaly audits
- CommunicationAlertAgent: SACHET early-warning & escalation
- ReportingAgent: Stage-3 multilingual narrative sit-reps & operator VQA
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from data_pipeline.schema import DisasterClass, GeoPoint
from models.stage1_classifier import DisasterTriageEngine
from models.stage2_severity import FloodSeverityHead, StructuralDamageHead
from models.stage3_reasoning_vlm import MultilingualDisasterVLMReasoner, OperationalSitRep
from models.multimodal_fusion import MultimodalFusionInterface
from analytics.impact_engine import ImpactNeedsEngine, ImpactAssessmentReport
from analytics.fund_allocation import SDRFFundAllocator, SDRFFundRequirementReport
from analytics.anomaly_detector import DamageAnomalyDetector
from orchestration.hitl_gate import CommandCenterHITLGate, GatedActionType
from live_monitoring.alert_service import TypedAlertEvent


class DetectionAgent:
    """Handles alert intake and Stage-1 triage."""
    def __init__(self):
        self.triage_engine = DisasterTriageEngine()

    def process_frame(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        return self.triage_engine.predict(frame_rgb)


class AssessmentAgent:
    """Handles Stage-2 deep damage analysis and cloud-cover SAR fusion."""
    def __init__(self):
        self.flood_head = FloodSeverityHead()
        self.structural_head = StructuralDamageHead()
        self.multimodal_sar = MultimodalFusionInterface()

    def assess(
        self, 
        disaster_class: DisasterClass, 
        image_rgb: np.ndarray,
        sar_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        if disaster_class == DisasterClass.FLOOD:
            return self.flood_head.analyze(image_rgb)
        elif disaster_class in [DisasterClass.EARTHQUAKE_COLLAPSE, DisasterClass.CYCLONE_STORM]:
            return self.multimodal_sar.assess_damage_all_weather(image_rgb, sar_data)
        return {"disaster_class": disaster_class.value, "status": "standard_evaluation"}


class ResourceFundsAgent:
    """Handles population impact modeling and SDRF fund estimations."""
    def __init__(self):
        self.impact_engine = ImpactNeedsEngine()
        self.fund_allocator = SDRFFundAllocator()
        self.anomaly_detector = DamageAnomalyDetector()

    def evaluate(
        self,
        zone_id: str,
        state_code: str,
        stage1_out: Dict[str, Any],
        stage2_out: Dict[str, Any]
    ) -> Tuple[ImpactAssessmentReport, SDRFFundRequirementReport]:
        impact = self.impact_engine.compute_impact(zone_id, stage1_out, stage2_out)
        funds = self.fund_allocator.estimate_relief_funds(
            zone_id, state_code, stage2_out, impact.estimated_people_exposed
        )
        return impact, funds


class ReportingAgent:
    """Handles advisory situation reports and multilingual communications."""
    def __init__(self):
        self.reasoner = MultilingualDisasterVLMReasoner()

    def generate_report(
        self,
        incident_id: str,
        stage1_out: Dict[str, Any],
        stage2_out: Dict[str, Any],
        language: str = "en"
    ) -> OperationalSitRep:
        return self.reasoner.generate_incident_report(incident_id, stage1_out, stage2_out, language)


class MasterOrchestrator:
    """Coordinates the specialized agents and mediates with the Command Center HITL Gate."""

    def __init__(self, hitl_gate: Optional[CommandCenterHITLGate] = None):
        self.detection = DetectionAgent()
        self.assessment = AssessmentAgent()
        self.resources = ResourceFundsAgent()
        self.reporting = ReportingAgent()
        self.hitl_gate = hitl_gate or CommandCenterHITLGate()

    def run_recon_and_assessment_cycle(
        self,
        incident_id: str,
        zone_id: str,
        state_code: str,
        aerial_frame_rgb: np.ndarray,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Executes full multi-agent analytical pipeline."""
        # 1. Detection
        stage1_res = self.detection.process_frame(aerial_frame_rgb)
        dclass = stage1_res["disaster_class"]

        # 2. Assessment
        stage2_res = self.assessment.assess(dclass, aerial_frame_rgb)

        # 3. Resources & Fund Estimation
        impact_rep, fund_rep = self.resources.evaluate(zone_id, state_code, stage1_res, stage2_res)

        # 4. Multilingual Sit-Rep
        sitrep = self.reporting.generate_report(incident_id, stage1_res, stage2_res, language)

        # 5. Propose Fund Allocation to Human Gate (Mandatory)
        approval_req = self.hitl_gate.submit_for_approval(
            action_type=GatedActionType.FUND_RELEASE,
            initiating_agent="ResourceFundsAgent",
            target_zone_id=zone_id,
            summary=f"Statutory SDRF fund estimate ₹{fund_rep.total_estimated_fund_inr:,.2f} for {zone_id}",
            payload=fund_rep.model_dump()
        )

        return {
            "incident_id": incident_id,
            "stage1_triage": stage1_res,
            "stage2_assessment": stage2_res,
            "impact_report": impact_rep.model_dump(),
            "fund_report": fund_rep.model_dump(),
            "sitrep": sitrep.model_dump(),
            "hitl_pending_approval": approval_req.model_dump()
        }

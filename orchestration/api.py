"""FastAPI REST API Service for Command Center Dashboards and Downstream Integration.
Exposes modular endpoints for alert ingestion, computer vision inference,
drone mission generation, HITL human reviews, and cryptographic audit log verification.
"""

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np
import base64
import cv2

from config.version import __version__
from data_pipeline.schema import DisasterClass, GeoPoint
from live_monitoring.alert_service import LiveMonitoringService, TypedAlertEvent
from drone_abstraction.protocol import DroneMissionContract
from drone_abstraction.dgca_compliance import DGCAComplianceChecker
from orchestration.agents import MasterOrchestrator
from orchestration.hitl_gate import CommandCenterHITLGate, GatedActionType
from orchestration.audit_log import CryptographicAuditLog
from models.registry import ModelRegistry

app = FastAPI(
    title="India Disaster Response AI Platform API",
    version=__version__,
    description="Modular AI system for disaster classification, drone orchestration, and SDRF fund analytics"
)

# Global Singletons
audit_log = CryptographicAuditLog()
hitl_gate = CommandCenterHITLGate(audit_log=audit_log)
orchestrator = MasterOrchestrator(hitl_gate=hitl_gate)
monitoring_service = LiveMonitoringService()
dgca_checker = DGCAComplianceChecker()
model_registry = ModelRegistry()


class AlertIngestRequest(BaseModel):
    cap_xml_or_json: str


class MissionPlanRequest(BaseModel):
    mission_id: str
    target_zone_id: str
    center_gps: GeoPoint
    grid_radius_m: float = 500.0
    altitude_agl_m: float = 60.0
    npnt_token: str = "VALID_DGCA_NPNT_DIGITAL_SKY_TOKEN_2026"


class HITLReviewRequest(BaseModel):
    request_id: str
    approve: bool
    operator_id: str
    operator_signature: str
    comments: str = "Operational sign-off granted"


@app.get("/api/v1/system/status")
def get_system_status():
    """System health check and active model registry pointers."""
    return {
        "status": "operational",
        "platform_version": __version__,
        "active_models": model_registry.active_models,
        "pending_hitl_approvals": len(hitl_gate.list_pending()),
        "supported_languages": ["en", "hi", "or", "ml", "bn", "ta", "te"]
    }


@app.post("/api/v1/alerts/ingest")
def ingest_cap_alert(req: AlertIngestRequest):
    """Parses SACHET/CAP alert, tests geofence, and triggers internal typed alert if relevant."""
    event = monitoring_service.process_incoming_cap(req.cap_xml_or_json)
    if not event:
        return {"status": "ignored", "reason": "Alert outside monitored operational radius or duplicate"}
    return {"status": "alert_raised", "event": event.model_dump()}


@app.post("/api/v1/missions/plan")
def plan_drone_mission(req: MissionPlanRequest):
    """Generates a MAVLink mission contract and audits against DGCA Drone Rules 2021."""
    mission = DroneMissionContract.create_grid_survey_mission(
        mission_id=req.mission_id,
        zone_id=req.target_zone_id,
        center_gps=req.center_gps,
        grid_radius_m=req.grid_radius_m,
        altitude_agl_m=req.altitude_agl_m
    )
    is_cleared, violations, audit = dgca_checker.verify_mission(mission, req.npnt_token)
    if not is_cleared:
        raise HTTPException(
            status_code=403,
            detail={"error": "DGCA Airspace Violation", "violations": violations, "audit": audit}
        )

    # Submit to HITL gate before physical takeoff
    approval = hitl_gate.submit_for_approval(
        action_type=GatedActionType.MISSION_DISPATCH,
        initiating_agent="MissionPlanner",
        target_zone_id=req.target_zone_id,
        summary=f"Automated grid survey mission {req.mission_id} at {req.altitude_agl_m}m AGL",
        payload=mission.model_dump()
    )

    return {
        "mission_contract": mission.model_dump(),
        "dgca_clearance": audit,
        "hitl_approval_required": approval.model_dump()
    }


@app.post("/api/v1/hitl/review")
def review_hitl_action(req: HITLReviewRequest):
    """Allows Command Center officers to approve or reject pending irreversible actions."""
    try:
        updated = hitl_gate.review_request(
            request_id=req.request_id,
            approve=req.approve,
            operator_id=req.operator_id,
            operator_signature=req.operator_signature,
            comments=req.comments
        )
        return {"status": "success", "request": updated.model_dump()}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/hitl/pending")
def list_pending_approvals():
    return {"pending_requests": [r.model_dump() for r in hitl_gate.list_pending()]}


@app.get("/api/v1/audit/verify")
def verify_audit_ledger():
    """Validates the cryptographic hash-chain integrity of the append-only audit trail."""
    is_intact, count, message = audit_log.verify_integrity()
    return {
        "is_intact": is_intact,
        "blocks_audited": count,
        "verification_summary": message
    }


class ModelRollbackRequest(BaseModel):
    stage: str
    target: str  # version string (e.g. '1.1.0') or model_id (e.g. 'stage1_mobilenetv3_triage_v1')
    operator_id: str
    reason: str = "Rollback requested by command center operator"


class ModelActivateRequest(BaseModel):
    model_id: str
    operator_id: str
    reason: str = "Model activation requested by command center operator"


@app.get("/api/v1/models/versions")
def list_model_versions(stage: Optional[str] = None):
    """Enumerates registered model artifact versions, active status, metrics, and checksums."""
    versions = model_registry.list_versions(stage=stage)
    return {
        "platform_version": __version__,
        "stage_filter": stage,
        "total_versions": len(versions),
        "versions": versions
    }


@app.get("/api/v1/models/compare")
def compare_model_versions(stage: str):
    """Compares registered versions and metrics for a specific pipeline stage."""
    comparison = model_registry.get_version_comparison(stage=stage)
    if comparison["num_versions"] == 0:
        raise HTTPException(status_code=404, detail=f"No models registered for stage '{stage}'")
    return comparison


@app.post("/api/v1/models/activate")
def activate_model_endpoint(req: ModelActivateRequest):
    """Activates a designated model artifact and records a cryptographic audit entry."""
    try:
        activated = model_registry.activate_model(req.model_id)
        audit_log.append_block(
            event_type="MODEL_ACTIVATION",
            actor=req.operator_id,
            details={
                "model_id": activated.model_id,
                "version": activated.version,
                "stage": activated.stage,
                "sha256": activated.sha256_checksum,
                "reason": req.reason
            }
        )
        return {
            "status": "activated",
            "active_model": activated.model_dump(),
            "operator_id": req.operator_id
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/models/rollback")
def rollback_model_stage(req: ModelRollbackRequest):
    """Rolls back the active model for a stage to a prior version or artifact ID with audit log."""
    prev_active = model_registry.get_active_model(req.stage)
    rolled_back = model_registry.rollback(stage=req.stage, target=req.target)
    if not rolled_back:
        raise HTTPException(
            status_code=404,
            detail=f"No matching model found for stage '{req.stage}' with target '{req.target}'"
        )
    audit_log.append_block(
        event_type="MODEL_ROLLBACK",
        actor=req.operator_id,
        details={
            "stage": req.stage,
            "target": req.target,
            "previous_model_id": prev_active.model_id if prev_active else None,
            "previous_version": prev_active.version if prev_active else None,
            "new_model_id": rolled_back.model_id,
            "new_version": rolled_back.version,
            "new_sha256": rolled_back.sha256_checksum,
            "reason": req.reason
        }
    )
    return {
        "status": "rolled_back",
        "stage": req.stage,
        "active_model": rolled_back.model_dump(),
        "operator_id": req.operator_id
    }


@app.get("/api/v1/models/verify")
def verify_model_checksums(stage: Optional[str] = None):
    """Verifies that registered model weights exist on disk and match their SHA-256 checksums."""
    report = model_registry.verify_checksums(stage=stage)
    all_valid = all(v.get("valid", False) for v in report.values())
    return {
        "all_valid": all_valid,
        "models_checked": len(report),
        "results": report
    }


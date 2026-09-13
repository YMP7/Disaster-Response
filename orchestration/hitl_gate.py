"""Human-in-the-Loop (HITL) Safety & Command Approval Gate.
Enforces the mandatory architectural boundary: NO irreversible action
(fund release, supply drop, drone mission launch, responder redirection)
can execute without explicit, cryptographically logged human authorization.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import time
from pydantic import BaseModel, Field
from orchestration.audit_log import CryptographicAuditLog


class GatedActionType(str, Enum):
    MISSION_DISPATCH = "mission_dispatch"
    SUPPLY_DROP = "supply_drop"
    RESPONDER_REDIRECTION = "responder_redirection"
    FUND_RELEASE = "fund_release"


class PendingApprovalRequest(BaseModel):
    request_id: str
    action_type: GatedActionType
    initiating_agent: str
    target_zone_id: str
    summary: str
    payload: Dict[str, Any]
    created_at: float
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[float] = None
    review_comments: Optional[str] = None


class CommandCenterHITLGate:
    """Safety-critical approval gate protecting physical and financial assets."""

    def __init__(self, audit_log: Optional[CryptographicAuditLog] = None):
        self.audit_log = audit_log or CryptographicAuditLog()
        self.pending_queue: Dict[str, PendingApprovalRequest] = {}

    def submit_for_approval(
        self,
        action_type: GatedActionType,
        initiating_agent: str,
        target_zone_id: str,
        summary: str,
        payload: Dict[str, Any]
    ) -> PendingApprovalRequest:
        req_id = f"REQ_{action_type.value.upper()}_{int(time.time() * 1000)}"
        req = PendingApprovalRequest(
            request_id=req_id,
            action_type=action_type,
            initiating_agent=initiating_agent,
            target_zone_id=target_zone_id,
            summary=summary,
            payload=payload,
            created_at=time.time()
        )
        self.pending_queue[req_id] = req

        # Record submission in audit log
        self.audit_log.record_event(
            actor_id=initiating_agent,
            action_type=f"HITL_PROPOSAL_{action_type.value.upper()}",
            payload={"request_id": req_id, "summary": summary, "zone_id": target_zone_id}
        )

        return req

    def review_request(
        self,
        request_id: str,
        approve: bool,
        operator_id: str,
        operator_signature: str,
        comments: str = "Approved per tactical review."
    ) -> PendingApprovalRequest:
        """Processes human operator command decision."""
        if request_id not in self.pending_queue:
            raise KeyError(f"Request {request_id} not found in pending HITL queue")

        req = self.pending_queue[request_id]
        req.status = "APPROVED" if approve else "REJECTED"
        req.reviewed_by = operator_id
        req.review_timestamp = time.time()
        req.review_comments = comments

        # Persist to cryptographic audit trail
        self.audit_log.record_event(
            actor_id=operator_id,
            action_type=f"HITL_{req.status}_{req.action_type.value.upper()}",
            payload={
                "request_id": request_id,
                "status": req.status,
                "comments": comments,
                "action_payload": req.payload
            },
            operator_signature=operator_signature
        )

        return req

    def list_pending(self) -> List[PendingApprovalRequest]:
        return [r for r in self.pending_queue.values() if r.status == "PENDING"]

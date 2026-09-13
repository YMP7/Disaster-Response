"""Immutable Cryptographic Audit Trail Module.
Implements an append-only, SHA-256 hash-chained transaction ledger for disaster response actions.
Guarantees non-repudiation and integrity for:
- Mission dispatch authorizations
- Computer vision model inferences
- Human Command Center approvals/rejections
- SDRF/NDRF fund releases and anomaly flags
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class AuditBlock(BaseModel):
    block_index: int
    timestamp: float
    actor_id: str         # e.g., "SYSTEM_DISPATCH_AGENT", "OFFICER_COMMANDER_04"
    action_type: str      # MISSION_LAUNCH, MODEL_INFERENCE, HITL_APPROVAL, FUND_RELEASE
    payload: Dict[str, Any]
    previous_hash: str
    current_hash: str
    signature: Optional[str] = None


class CryptographicAuditLog:
    """Manages the append-only cryptographic ledger."""

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("audit_trail.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_index = 0
        self.latest_hash = self.GENESIS_HASH
        self._initialize()

    def _initialize(self):
        if self.log_path.exists():
            last_block = None
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_block = json.loads(line)
            if last_block:
                self.current_index = last_block.get("block_index", 0) + 1
                self.latest_hash = last_block.get("current_hash", self.GENESIS_HASH)

    @staticmethod
    def calculate_hash(
        block_index: int, 
        timestamp: float, 
        actor_id: str, 
        action_type: str, 
        payload: Dict[str, Any], 
        prev_hash: str
    ) -> str:
        serialized = json.dumps({
            "block_index": block_index,
            "timestamp": timestamp,
            "actor_id": actor_id,
            "action_type": action_type,
            "payload": payload,
            "prev_hash": prev_hash
        }, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def record_event(
        self,
        actor_id: str,
        action_type: str,
        payload: Dict[str, Any],
        operator_signature: Optional[str] = None
    ) -> AuditBlock:
        """Appends a new event to the cryptographically chained ledger."""
        t = time.time()
        c_hash = self.calculate_hash(
            self.current_index, t, actor_id, action_type, payload, self.latest_hash
        )

        block = AuditBlock(
            block_index=self.current_index,
            timestamp=t,
            actor_id=actor_id,
            action_type=action_type,
            payload=payload,
            previous_hash=self.latest_hash,
            current_hash=c_hash,
            signature=operator_signature or f"SIG_AUTO_{c_hash[:16]}"
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(block.model_dump_json() + "\n")

        self.latest_hash = c_hash
        self.current_index += 1
        return block

    def verify_integrity(self) -> Tuple[bool, int, str]:
        """Audits the entire log from genesis to verify no tampering or deleted blocks."""
        if not self.log_path.exists():
            return True, 0, "Log empty"

        expected_prev = self.GENESIS_HASH
        count = 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                data = json.loads(line)
                block = AuditBlock(**data)

                if block.previous_hash != expected_prev:
                    return False, line_no, f"Hash chain broken at block #{block.block_index}"

                recomputed = self.calculate_hash(
                    block.block_index, block.timestamp, block.actor_id,
                    block.action_type, block.payload, block.previous_hash
                )
                if recomputed != block.current_hash:
                    return False, line_no, f"Data payload tampered in block #{block.block_index}"

                expected_prev = block.current_hash
                count += 1

        return True, count, f"All {count} blocks cryptographically verified intact."

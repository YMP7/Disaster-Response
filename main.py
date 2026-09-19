"""End-to-End Incident Lifecycle Demonstration.
Orchestrates the entire platform workflow:
1. SACHET CAP Alert Ingestion (Odisha Cyclone & Flood)
2. Geofence Radius Spatial Match
3. DGCA Drone Rules 2021 Aviation Clearance
4. MAVLink Mission Contract Formulation
5. PX4 SITL Simulator Autonomous Flight
6. On-Scene Edge CV Triage & Severity Assessment
7. DPDP Automated Privacy Redaction
8. WorldPop Population Exposure & Needs Vector
9. SDRF/NDRF Statutory Fund Calculation with Explainable Justification
10. Municipal Claim Anomaly & Fraud Audit
11. Stage-3 Multilingual Advisory Sit-Rep (English & Odia)
12. Command Center Human Approval Gate & Cryptographic Audit Verification
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(legacy_windows=False, force_terminal=True)
from live_monitoring.alert_service import LiveMonitoringService
from drone_abstraction.protocol import DroneMissionContract
from drone_abstraction.dgca_compliance import DGCAComplianceChecker
from drone_abstraction.simulator import DroneSimulatorHarness
from drone_abstraction.on_scene_agent import OnSceneDroneAgent, DroneActionType
from data_pipeline.privacy_filter import DPDPPrivacyFilter
from orchestration.agents import MasterOrchestrator
from orchestration.hitl_gate import CommandCenterHITLGate, GatedActionType
from orchestration.audit_log import CryptographicAuditLog
from analytics.anomaly_detector import DamageAnomalyDetector

console = Console()


def run_full_incident_simulation():
    console.print(Panel.fit(
        "[bold cyan]INDIA DISASTER RESPONSE & AUTONOMOUS DRONE PLATFORM[/bold cyan]\n"
        "[dim]Safety-Critical, Simulator-First AI Platform (Grounded in MHA SDRF Norms & DGCA Rules 2021)[/dim]",
        border_style="cyan"
    ))

    # Initialize Platform Infrastructure
    audit_log = CryptographicAuditLog(log_path=Path("audit_trail.jsonl"))
    hitl_gate = CommandCenterHITLGate(audit_log=audit_log)
    orchestrator = MasterOrchestrator(hitl_gate=hitl_gate)
    monitoring = LiveMonitoringService()
    dgca = DGCAComplianceChecker()
    privacy = DPDPPrivacyFilter()
    anomaly_detector = DamageAnomalyDetector()

    # Step 1: Ingest Live SACHET / CAP Alert Feed
    console.print("\n[bold yellow]═══ STEP 1: PUBLIC FEED INGESTION & GEOFENCING ═══[/bold yellow]")
    xml_alert = LiveMonitoringService.get_sample_sachet_odisha_cyclone_alert()
    alert_event = monitoring.process_incoming_cap(xml_alert)
    console.print(f"✓ Ingested SACHET CAP v1.2 Alert: [bold]{alert_event.event_id}[/bold]")
    console.print(f"  • Source: {alert_event.source}")
    console.print(f"  • Event: [bold red]{alert_event.headline}[/bold red]")
    console.print(f"  • Geofence Intersect: Matched [green]{alert_event.affected_zone_name}[/green] within {alert_event.radius_km} km radius")
    console.print(f"  • Drone Reconnaissance Mandated: [bold]{alert_event.requires_drone_recon}[/bold]")

    # Step 2: Mission Planning & DGCA Compliance Audit
    console.print("\n[bold yellow]═══ STEP 2: DGCA AIRSPACE & MAVLINK MISSION PLANNING ═══[/bold yellow]")
    mission = DroneMissionContract.create_grid_survey_mission(
        mission_id="ODISHA-RECON-TASK-01",
        zone_id=alert_event.affected_zone_id,
        center_gps=alert_event.epicenter_gps,
        grid_radius_m=600.0,
        altitude_agl_m=65.0  # Safe altitude below 120m ceiling
    )
    cleared, violations, dgca_audit = dgca.verify_mission(mission)
    console.print(f"✓ DGCA Drone Rules 2021 Compliance: [bold green]{'CLEARED' if cleared else 'REJECTED'}[/bold green]")
    console.print(f"  • Airspace Classification: [green]{dgca_audit['airspace_classification']} ZONE[/green]")
    console.print(f"  • Operating Altitude: {mission.cruise_altitude_agl_m}m AGL (Statutory Limit: 120m AGL)")
    console.print(f"  • NPNT Token Verified: {dgca_audit['npnt_verified']}")
    console.print(f"  • Remote Pilot in Command: {dgca_audit['rpic_id']}")

    # Step 3: PX4 SITL Flight Simulation & Live Camera Stream
    console.print("\n[bold yellow]═══ STEP 3: PX4 SITL FLIGHT SIMULATION & LIVE ON-SCENE AGENT ═══[/bold yellow]")
    sim = DroneSimulatorHarness(synthetic_disaster=alert_event.disaster_class)
    drone_agent = OnSceneDroneAgent()

    survey_frames = []
    agent_decisions = []

    console.print("  [dim]Drone executing automated MAVLink lawnmower grid... streaming telemetry & frames...[/dim]")
    for telemetry, frame_rgb, status in sim.execute_mission_stream(mission):
        # Apply DPDP automated privacy blurring on incoming feed
        redacted_frame, priv_meta = privacy.redact_pii(frame_rgb)
        survey_frames.append(redacted_frame)
        
        decision = drone_agent.evaluate_live_frame(redacted_frame, telemetry, status)
        agent_decisions.append(decision)
        console.print(f"    → Waypoint #{status['current_waypoint']} ({telemetry.altitude_agl_m:.1f}m AGL | Bat: {status['battery_pct']}%) | Action: [bold cyan]{decision.action.value.upper()}[/bold cyan]")

    last_decision = agent_decisions[-2] if len(agent_decisions) > 1 else agent_decisions[0]
    console.print(f"✓ On-Scene Decision Agent Flagged: [bold red]{last_decision.action.value.upper()}[/bold red]")
    console.print(f"  • Justification: {last_decision.justification}")
    console.print(f"  • DPDP Act Compliance: PII Scrubbed ({priv_meta['total_pii_anonymized']} faces/plates masked)")

    # Step 4: Multi-Agent Quantitative Assessment & Fund Analytics
    console.print("\n[bold yellow]═══ STEP 4: MULTI-AGENT ASSESSMENT & EXPLAINABLE SDRF FUNDS ═══[/bold yellow]")
    analysis = orchestrator.run_recon_and_assessment_cycle(
        incident_id="INC-ODISHA-FLOOD-2026",
        zone_id=alert_event.affected_zone_id,
        state_code="OD",
        aerial_frame_rgb=survey_frames[-1],
        language="en"
    )

    stage1 = analysis["stage1_triage"]
    stage2 = analysis["stage2_assessment"]
    impact = analysis["impact_report"]
    fund = analysis["fund_report"]

    console.print(f"✓ Stage-1 Edge Classifier: [bold green]{stage1['disaster_class'].upper()}[/bold green] (Conf: {stage1['confidence']*100:.1f}%)")
    console.print(f"✓ Stage-2 Flood Extent: [bold]{stage2.get('water_extent_percentage', 0.0)}% surface inundation[/bold]")
    console.print(f"  • Submerged Housing Units: {stage2.get('buildings_flooded', 0)}")
    console.print(f"  • Corridors Status: [bold red]{stage2.get('road_passability', 'clear').upper()}[/bold red]")
    console.print(f"  • Population Exposure: [bold]{impact['estimated_people_exposed']:,} citizens[/bold]")
    console.print(f"  • Primary Urgency Driver: [bold red]{impact['primary_urgency_driver']}[/bold red]")

    # Print Itemized SDRF Table
    table = Table(title=f"Statutory SDRF/NDRF Relief Requirement (MHA Norms) — Total: ₹{fund['total_estimated_fund_inr']:,.2f} ({fund['total_estimated_fund_crores']} Crores)")
    table.add_column("Category", style="cyan")
    table.add_column("Item Description", style="white")
    table.add_column("Qty", justify="right")
    table.add_column("Unit Rate (INR)", justify="right")
    table.add_column("Total (INR)", justify="right", style="green")
    table.add_column("MHA Schedule Ref", style="dim")

    for item in fund["line_items"]:
        table.add_row(
            item["category"],
            item["unit_description"],
            str(item["quantity"]),
            f"₹{item['unit_rate_inr']:,.2f}",
            f"₹{item['total_amount_inr']:,.2f}",
            item["mha_norm_reference"]
        )
    console.print(table)

    console.print(Panel(fund["plain_language_explanation"], title="[bold]Explainable Financial Attribution[/bold]", border_style="green"))

    # Step 5: Fraud & Anomaly Audit
    console.print("\n[bold yellow]═══ STEP 5: CLAIM ANOMALY & FRAUD DETECTION AUDIT ═══[/bold yellow]")
    simulated_claim_amount = 7_500_000.0  # Inflated municipal claim
    audit_check = anomaly_detector.audit_claim(
        claim_id="MUNICIPAL-CLAIM-CUTTACK-09",
        zone_id=alert_event.affected_zone_id,
        claimed_houses=45,
        claimed_amount_inr=simulated_claim_amount,
        imagery_verified_houses=stage2.get("buildings_flooded", 2),
        imagery_verified_amount_inr=fund["total_estimated_fund_inr"]
    )
    console.print(f"  • Self-Reported Claim: ₹{simulated_claim_amount:,.2f} | Aerial Verified: ₹{fund['total_estimated_fund_inr']:,.2f}")
    console.print(f"  • Discrepancy Ratio: [bold red]{audit_check.discrepancy_ratio}x[/bold red]")
    console.print(f"  • Anomaly Status: [bold red]{audit_check.risk_level} RISK[/bold red]")
    console.print(f"  • Audit Directive: {audit_check.audit_recommendation}")

    # Step 6: Multilingual Advisory Situation Reports (SACHET-Aligned)
    console.print("\n[bold yellow]═══ STEP 6: MULTILINGUAL ADVISORY SIT-REPS (SACHET-ALIGNED) ═══[/bold yellow]")
    sitrep_en = analysis["sitrep"]
    sitrep_or = orchestrator.reporting.generate_report(
        "INC-ODISHA-FLOOD-2026", stage1, stage2, language="or"
    )
    console.print(f"[bold cyan]English Sit-Rep:[/bold cyan] {sitrep_en['headline']}")
    console.print(f"  {sitrep_en['executive_summary']}")
    console.print(f"[bold cyan]Odia Sit-Rep:[/bold cyan] {sitrep_or.headline}")
    console.print(f"  {sitrep_or.executive_summary}")

    # Step 7: Safety-Critical Human-in-the-Loop Command Gate
    console.print("\n[bold yellow]═══ STEP 7: HUMAN COMMAND CENTER GATE & CRYPTOGRAPHIC AUDIT LOG ═══[/bold yellow]")
    pending = hitl_gate.list_pending()
    console.print(f"  [bold]Pending Gated Approvals in Queue:[/bold] {len(pending)}")
    for p in pending:
        console.print(f"    • [{p.status}] {p.request_id}: {p.summary}")

    # Officer cryptographically signs and authorizes
    req_to_approve = pending[0]
    operator_id = "OFFICER_PATNAIK_SRC_ODISHA"
    ed25519_sig = hitl_gate.sign_request(
        request_id=req_to_approve.request_id,
        approve=True,
        operator_id=operator_id
    )
    reviewed = hitl_gate.review_request(
        request_id=req_to_approve.request_id,
        approve=True,
        operator_id=operator_id,
        operator_signature=ed25519_sig,
        comments="Damage and SDRF schedule verified against aerial computer vision. Authorized for initial tranche release."
    )
    console.print(f"✓ Officer Action Executed: [bold green]{reviewed.status}[/bold green] by {reviewed.reviewed_by}")
    console.print(f"  [dim]Ed25519 Digital Signature: {reviewed.operator_signature[:24]}... (64 bytes)[/dim]")
    console.print(f"  [dim]Verified Public Key:       {reviewed.operator_public_key[:24]}...[/dim]")

    # Verify Cryptographic Hash Chain Integrity
    is_intact, blocks_audited, msg = audit_log.verify_integrity()
    console.print(f"✓ Cryptographic Ledger Audit: [bold green]{msg}[/bold green] ({blocks_audited} chained SHA-256 blocks)")

    console.print(Panel.fit(
        "[bold green]✓ END-TO-END DISASTER INCIDENT SIMULATION COMPLETE[/bold green]\n"
        "[dim]Zero Physical Hardware Used | 100% Protocol-Agnostic MAVLink | Fully Audited[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    run_full_incident_simulation()

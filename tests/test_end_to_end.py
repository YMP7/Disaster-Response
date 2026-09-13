"""Comprehensive End-to-End Automated Test Suite.
Verifies all 8 deliverables:
1. Schemas & Ingestion
2. India Domain Augmentation & DPDP Privacy Filter
3. Stage-1 Edge Triage, Calibration, & Gating
4. Stage-2 Severity Heads & Multimodal SAR Fusion
5. Stage-3 Multilingual Advisory VLM Reporting
6. Real India Held-Out Benchmark Suite
7. SACHET CAP Parser & Radius Geofence
8. DGCA Drone Rules 2021 & MAVLink Mission Protocol
9. Drone Flight Simulator & On-Scene Decision Agent
10. Impact Needs & SDRF/NDRF Fund Math with Explainability
11. Anomaly / Claim Fraud Detector
12. Cryptographic Audit Log & HITL Safety Gate
13. Extensibility Plugin Registration
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Core imports
from data_pipeline.schema import (
    DisasterClass, DamageGrade, RoadPassability, BoundingBox, GeoPoint
)
from data_pipeline.india_augmentation import IndiaDomainAugmentor
from data_pipeline.privacy_filter import DPDPPrivacyFilter
from data_pipeline.synthetic_generator import SyntheticAerialGenerator
from data_pipeline.india_curated_set import IndiaCuratedBenchmark

from models.stage1_classifier import Stage1EdgeClassifier, DisasterTriageEngine
from models.stage2_severity import FloodSeverityHead, StructuralDamageHead
from models.calibration import ReliabilityEvaluator
from models.stage3_reasoning_vlm import MultilingualDisasterVLMReasoner
from models.multimodal_fusion import MultimodalFusionInterface
from models.registry import ModelRegistry

from live_monitoring.cap_parser import SACHETCAPParser
from live_monitoring.geofence import GeofenceEngine, MonitoredZone
from live_monitoring.alert_service import LiveMonitoringService

from drone_abstraction.protocol import DroneMissionContract, MAVCmdType
from drone_abstraction.dgca_compliance import DGCAComplianceChecker
from drone_abstraction.simulator import DroneSimulatorHarness
from drone_abstraction.on_scene_agent import OnSceneDroneAgent, DroneActionType

from analytics.impact_engine import ImpactNeedsEngine
from analytics.fund_allocation import SDRFFundAllocator
from analytics.anomaly_detector import DamageAnomalyDetector

from orchestration.audit_log import CryptographicAuditLog
from orchestration.hitl_gate import CommandCenterHITLGate, GatedActionType
from orchestration.agents import MasterOrchestrator
from plugins.base import plugin_registry
import plugins.examples.industrial_chemical  # Triggers auto-registration


class TestDataPipelineAndPrivacy:
    def test_bounding_box_validation(self):
        bbox = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.5, xmax=0.6)
        assert bbox.area == pytest.approx(0.16, 0.001)

    def test_india_domain_augmentation(self):
        img = np.full((128, 128, 3), 100, dtype=np.uint8)
        augmented = IndiaDomainAugmentor.augment_for_india(img, is_monsoon=True, is_coastal_cyclone=True)
        assert augmented.shape == (128, 128, 3)
        assert not np.array_equal(img, augmented)

    def test_dpdp_privacy_filter(self):
        filter_dpdp = DPDPPrivacyFilter()
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        redacted, audit = filter_dpdp.redact_pii(canvas)
        assert audit["dpdp_compliant"] is True
        assert redacted.shape == canvas.shape

    def test_synthetic_generator(self):
        gen = SyntheticAerialGenerator(image_size=(256, 256))
        img, anno = gen.generate_scene(disaster_type=DisasterClass.FLOOD)
        assert img.shape == (256, 256, 3)
        assert anno.disaster_class == DisasterClass.FLOOD
        assert len(anno.buildings) > 0


class TestComputerVisionAndFusion:
    def test_stage1_edge_triage(self):
        engine = DisasterTriageEngine()
        test_img = np.full((224, 224, 3), 120, dtype=np.uint8)
        res = engine.predict(test_img)
        assert "disaster_class" in res
        assert "confidence" in res
        assert "class_probabilities" in res

    def test_stage2_flood_severity(self):
        head = FloodSeverityHead()
        test_img = np.full((256, 256, 3), 100, dtype=np.uint8)
        # Paint silt water across bottom half (HSV in silt/blue range)
        cv2.rectangle(test_img, (0, 100), (256, 256), (30, 90, 180), -1)
        res = head.analyze(test_img)
        assert res["disaster_type"] == "flood"
        assert res["water_extent_percentage"] > 20.0
        assert "road_passability" in res

    def test_multimodal_sar_fusion(self):
        fusion = MultimodalFusionInterface()
        opt = np.full((256, 256, 3), 150, dtype=np.uint8)
        res = fusion.assess_damage_all_weather(opt)
        assert "damage_grade_distribution" in res
        assert "radar_reliance_factor" in res

    def test_calibration_ece_metrics(self):
        confs = np.array([0.9, 0.8, 0.7, 0.6])
        preds = np.array([1, 1, 0, 1])
        labels = np.array([1, 1, 1, 0])
        stats = ReliabilityEvaluator.compute_ece(confs, preds, labels)
        assert "expected_calibration_error" in stats
        assert stats["expected_calibration_error"] >= 0.0

    def test_multilingual_vlm_reporting(self):
        reasoner = MultilingualDisasterVLMReasoner()
        s1 = {"disaster_class": DisasterClass.FLOOD}
        s2 = {"water_extent_percentage": 45.0, "buildings_detected": 10, "buildings_flooded": 4, "road_passability": "road_blocked"}

        # English
        rep_en = reasoner.generate_incident_report("inc_01", s1, s2, language="en")
        assert "CRITICAL INCIDENT" in rep_en.headline

        # Hindi
        rep_hi = reasoner.generate_incident_report("inc_01", s1, s2, language="hi")
        assert "आपदा चेतावनी" in rep_hi.headline

        # Odia
        rep_or = reasoner.generate_incident_report("inc_01", s1, s2, language="or")
        assert "ବିପର୍ଯ୍ୟୟ" in rep_or.headline

        # Q&A test
        ans = reasoner.answer_operator_question("Are main roads blocked?", s1, s2)
        assert "BLOCKED" in ans

    def test_held_out_india_benchmarks(self):
        benchmarks = IndiaCuratedBenchmark.get_all_benchmarks()
        assert len(benchmarks) >= 2
        kerala_img, kerala_anno = benchmarks[0]
        assert kerala_anno.disaster_class == DisasterClass.FLOOD
        assert kerala_anno.metadata["state"] == "Kerala"

        odisha_img, odisha_anno = benchmarks[1]
        assert odisha_anno.disaster_class == DisasterClass.CYCLONE_STORM
        assert odisha_anno.metadata["state"] == "Odisha"


class TestLiveMonitoringAndAviation:
    def test_cap_parser_and_geofence(self):
        xml_feed = LiveMonitoringService.get_sample_sachet_odisha_cyclone_alert()
        service = LiveMonitoringService()
        event = service.process_incoming_cap(xml_feed)
        assert event is not None
        assert event.disaster_class == DisasterClass.CYCLONE_STORM
        assert event.affected_zone_id == "odisha_cuttack"
        assert event.requires_drone_recon is True

    def test_dgca_ceiling_and_red_zone_enforcement(self):
        checker = DGCAComplianceChecker()
        
        # Test valid mission <= 120m
        center = GeoPoint(latitude=20.4625, longitude=85.8828)
        valid_mission = DroneMissionContract.create_grid_survey_mission(
            "m_valid", "odisha_cuttack", center, altitude_agl_m=60.0
        )
        cleared, violations, _ = checker.verify_mission(valid_mission)
        assert cleared is True
        assert len(violations) == 0

        # Test ceiling violation (> 120m AGL)
        high_mission = DroneMissionContract.create_grid_survey_mission(
            "m_high", "odisha_cuttack", center, altitude_agl_m=150.0
        )
        cleared, violations, _ = checker.verify_mission(high_mission)
        assert cleared is False
        assert any("exceeds mandatory 120m" in v for v in violations)

        # Test Red Zone prohibited airspace violation
        red_point = GeoPoint(latitude=20.2961, longitude=85.8245) # Secretariat / Military
        red_mission = DroneMissionContract.create_grid_survey_mission(
            "m_red", "odisha_red", red_point, altitude_agl_m=50.0
        )
        cleared, violations, _ = checker.verify_mission(red_mission)
        assert cleared is False
        assert any("RED ZONE" in v for v in violations)

    def test_drone_simulator_and_on_scene_agent(self):
        sim = DroneSimulatorHarness(synthetic_disaster=DisasterClass.FLOOD)
        agent = OnSceneDroneAgent()
        center = GeoPoint(latitude=20.4625, longitude=85.8828)
        mission = DroneMissionContract.create_grid_survey_mission(
            "m_sim", "odisha_cuttack", center, altitude_agl_m=60.0
        )

        decisions = []
        for telemetry, frame, status in sim.execute_mission_stream(mission):
            decision = agent.evaluate_live_frame(frame, telemetry, status)
            decisions.append(decision)
            assert isinstance(decision.action, DroneActionType)

        assert len(decisions) > 0


class TestAnalyticsAndGovernance:
    def test_impact_and_sdrf_fund_engine(self):
        impact_engine = ImpactNeedsEngine()
        fund_allocator = SDRFFundAllocator()

        s1_out = {"disaster_class": DisasterClass.FLOOD}
        s2_out = {
            "water_extent_percentage": 50.0,
            "buildings_detected": 10,
            "buildings_flooded": 5,
            "road_passability": "road_blocked"
        }

        impact = impact_engine.compute_impact("cuttack_sec_4", s1_out, s2_out)
        assert impact.estimated_people_exposed > 0
        assert impact.needs.evacuation_transport_urgency > 0.5

        funds = fund_allocator.estimate_relief_funds(
            "cuttack_sec_4", "OD", s2_out, impact.estimated_people_exposed
        )
        assert funds.total_estimated_fund_inr > 0
        assert len(funds.line_items) >= 3
        assert "MHA SDRF" in funds.line_items[0].mha_norm_reference

    def test_anomaly_fraud_detection(self):
        detector = DamageAnomalyDetector()
        # Test inflated claim (claimed 50 destroyed vs 5 verified)
        audit = detector.audit_claim(
            claim_id="clm_001",
            zone_id="cuttack_sec_4",
            claimed_houses=50,
            claimed_amount_inr=6_000_000.0,
            imagery_verified_houses=5,
            imagery_verified_amount_inr=1_200_000.0
        )
        assert audit.is_anomalous is True
        assert audit.risk_level == "CRITICAL"
        assert "HOLD DISBURSEMENT" in audit.audit_recommendation

    def test_cryptographic_audit_log_and_hitl_gate(self, tmp_path):
        log_file = tmp_path / "test_audit.jsonl"
        audit = CryptographicAuditLog(log_path=log_file)
        gate = CommandCenterHITLGate(audit_log=audit)

        # Propose mission launch
        req = gate.submit_for_approval(
            action_type=GatedActionType.MISSION_DISPATCH,
            initiating_agent="TestAgent",
            target_zone_id="zone_test",
            summary="Test drone takeoff",
            payload={"wp_count": 5}
        )
        assert req.status == "PENDING"
        assert len(gate.list_pending()) == 1

        # Officer signs off
        reviewed = gate.review_request(
            request_id=req.request_id,
            approve=True,
            operator_id="OFFICER_PATNAIK_01",
            operator_signature="SIG_ECDSA_VALIDATED_9942",
            comments="Clear weather, mission approved."
        )
        assert reviewed.status == "APPROVED"
        assert len(gate.list_pending()) == 0

        # Verify cryptographic chain integrity
        is_intact, count, msg = audit.verify_integrity()
        assert is_intact is True
        assert count == 2  # 1 proposal + 1 approval

    def test_extensibility_plugin_registry(self):
        plugin = plugin_registry.get_disaster_head("industrial_chemical_toxic_plume")
        assert plugin is not None
        test_img = np.full((100, 100, 3), 120, dtype=np.uint8)
        res = plugin.analyze_scene(test_img, {})
        assert "mandated_exclusion_zone_meters" in res

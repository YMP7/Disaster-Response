"""Integration Test Suite for Real MAVLink v2.0 UDP Socket Communication.

SCOPE CAVEAT:
verified: correct MAVLink 2.0 protocol implementation over real UDP socket;
not yet verified: compatibility with actual PX4 Autopilot firmware.

Tests full-duplex binary MAVLink 2.0 framing, mission upload handshake,
flight commands (ARM, START, RTL), live telemetry decoding, and lossy network resilience.
All tests use real OS UDP sockets on dynamic ephemeral loopback ports.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import socket
import time
import pytest
from pymavlink import mavutil

from data_pipeline.schema import GeoPoint
from drone_abstraction.protocol import DroneMissionContract, MissionWaypoint, MAVCmdType
from drone_abstraction.mavlink_client import MAVLinkDroneClient
from drone_abstraction.sitl_server import PX4SITLEmulator
from drone_abstraction.simulator import DroneSimulatorHarness


def get_ephemeral_udp_port() -> int:
    """Finds an available local UDP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def build_sample_mission(mission_id: str = "TEST-MISSION-SITL-01") -> DroneMissionContract:
    """Builds a realistic 6-waypoint grid survey mission."""
    center = GeoPoint(latitude=20.2961, longitude=85.8245, altitude_m=60.0)
    return DroneMissionContract.create_grid_survey_mission(
        mission_id=mission_id,
        zone_id="ZONE-ODISHA-COASTAL",
        center_gps=center,
        grid_radius_m=300.0,
        altitude_agl_m=50.0
    )


def test_mavlink_udp_connection_and_heartbeat():
    """Confirms bidirectional binary MAVLink 2.0 heartbeat exchange over real UDP socket."""
    port = get_ephemeral_udp_port()
    server = PX4SITLEmulator(target_host="127.0.0.1", target_port=port, sys_id=1, comp_id=1)
    client = MAVLinkDroneClient(
        connection_str=f"udpin:127.0.0.1:{port}",
        heartbeat_timeout_sec=3.0
    )

    try:
        server.start()
        connected = client.connect(timeout=3.0)
        assert connected is True, "Client failed to connect and receive heartbeat"
        assert client.target_system == 1
        assert client.target_component == 1
        assert client.autopilot_type == mavutil.mavlink.MAV_AUTOPILOT_PX4
        assert client.vehicle_type == mavutil.mavlink.MAV_TYPE_QUADROTOR
        assert client.is_heartbeat_healthy() is True

        # Send GCS heartbeat back
        client.send_heartbeat()
    finally:
        client.close()
        server.stop()


def test_mavlink_mission_upload_handshake():
    """Verifies MISSION_COUNT -> MISSION_REQUEST_INT -> MISSION_ITEM_INT -> MISSION_ACK protocol."""
    port = get_ephemeral_udp_port()
    server = PX4SITLEmulator(target_host="127.0.0.1", target_port=port)
    client = MAVLinkDroneClient(
        connection_str=f"udpin:127.0.0.1:{port}",
        mission_timeout_sec=2.0
    )

    try:
        server.start()
        assert client.connect(timeout=3.0) is True

        mission = build_sample_mission()
        success = client.upload_mission(mission)
        assert success is True, "Mission upload handshake failed"

        # Verify server stored the exact mission items
        assert len(server.stored_mission_items) == len(mission.waypoints)
        assert server.stored_mission_items[0]["command"] == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
        assert server.stored_mission_items[-1]["command"] == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH
    finally:
        client.close()
        server.stop()


def test_mavlink_arm_and_flight_telemetry():
    """Verifies vehicle arming, mission execution, and live telemetry decoding over UDP."""
    port = get_ephemeral_udp_port()
    server = PX4SITLEmulator(target_host="127.0.0.1", target_port=port)
    client = MAVLinkDroneClient(
        connection_str=f"udpin:127.0.0.1:{port}",
        mission_timeout_sec=2.0
    )

    try:
        server.start()
        assert client.connect(timeout=3.0) is True

        mission = build_sample_mission()
        assert client.upload_mission(mission) is True

        # Arm vehicle
        assert client.arm_vehicle(True) is True
        assert server.is_armed is True

        # Start mission
        assert client.start_mission() is True
        assert server.is_mission_active is True

        # Wait briefly for simulated physics and telemetry emission
        time.sleep(0.5)

        # Poll incoming MAVLink telemetry
        result = client.poll_telemetry(timeout=1.0)
        assert result is not None, "Failed to receive telemetry frame over UDP"
        frame, status = result

        assert frame.gps.latitude > 0.0
        assert frame.gps.longitude > 0.0
        assert status["is_armed"] is True
        assert status["battery_pct"] <= 100.0
        assert status["is_link_healthy"] is True

        # Disarm vehicle
        assert client.arm_vehicle(False) is True
        assert server.is_armed is False
    finally:
        client.close()
        server.stop()


def test_mavlink_return_to_launch_command():
    """Verifies execution of MAV_CMD_NAV_RETURN_TO_LAUNCH command."""
    port = get_ephemeral_udp_port()
    server = PX4SITLEmulator(target_host="127.0.0.1", target_port=port)
    client = MAVLinkDroneClient(
        connection_str=f"udpin:127.0.0.1:{port}"
    )

    try:
        server.start()
        assert client.connect(timeout=3.0) is True

        mission = build_sample_mission()
        assert client.upload_mission(mission) is True
        client.arm_vehicle(True)
        client.start_mission()

        # Command RTL
        assert client.return_to_launch() is True
        assert server.is_mission_active is False
        assert server.alt_agl_m == 0.0
    finally:
        client.close()
        server.stop()


def test_mavlink_lossy_network_resilience():
    """Simulates dropped UDP packet during mission upload and confirms client retry recovery."""
    port = get_ephemeral_udp_port()
    server = PX4SITLEmulator(target_host="127.0.0.1", target_port=port)
    client = MAVLinkDroneClient(
        connection_str=f"udpin:127.0.0.1:{port}",
        mission_timeout_sec=0.8,
        max_retries=3
    )

    try:
        server.start()
        assert client.connect(timeout=3.0) is True

        # Inject dropped MISSION_REQUEST_INT for waypoint seq=2
        server.drop_request_seq = 2
        server.has_dropped_request = False

        mission = build_sample_mission()
        # Client should encounter timeout on seq=2, retry the upload cycle, and succeed
        success = client.upload_mission(mission)
        assert success is True, "Client failed to recover from dropped packet"
        assert server.has_dropped_request is True, "Server did not trigger packet drop hook"
        assert len(server.stored_mission_items) == len(mission.waypoints)
    finally:
        client.close()
        server.stop()


def test_mavlink_heartbeat_timeout_detection():
    """Confirms client detects lost datalink when heartbeats cease."""
    port = get_ephemeral_udp_port()
    server = PX4SITLEmulator(target_host="127.0.0.1", target_port=port)
    client = MAVLinkDroneClient(
        connection_str=f"udpin:127.0.0.1:{port}",
        heartbeat_timeout_sec=0.5
    )

    try:
        server.start()
        assert client.connect(timeout=3.0) is True
        assert client.is_heartbeat_healthy() is True

        # Abruptly stop server (simulating telemetry loss / power loss)
        server.stop()
        time.sleep(0.7)

        assert client.is_heartbeat_healthy() is False, "Client failed to detect heartbeat timeout"
    finally:
        client.close()


def test_drone_simulator_harness_modes():
    """Verifies that DroneSimulatorHarness executes in both MOCK and SITL_UDP modes."""
    mission = build_sample_mission("TEST-MODES-01")

    # 1. Test Mock Mode (safe default)
    mock_harness = DroneSimulatorHarness(comm_mode="MOCK_HARNESS")
    mock_frames = list(mock_harness.execute_mission_stream(mission))
    assert len(mock_frames) == len(mission.waypoints)
    telem0, img0, status0 = mock_frames[0]
    assert status0["current_waypoint"] == 0

    # 2. Test SITL UDP Mode (real socket)
    port = get_ephemeral_udp_port()
    sitl_harness = DroneSimulatorHarness(comm_mode="SITL_UDP", udp_port=port)
    sitl_frames = list(sitl_harness.execute_mission_stream(mission))
    assert len(sitl_frames) == len(mission.waypoints)
    telem_sitl, img_sitl, status_sitl = sitl_frames[0]
    assert status_sitl["current_waypoint"] == 0

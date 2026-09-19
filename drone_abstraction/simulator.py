"""PX4 SITL-Compatible Autonomous Drone Flight Simulator.
Executes DroneMissionContract, emits MAVLink-style telemetry, simulates battery draw,
and synthesizes live aerial video frames at each waypoint for on-scene computer vision.

Supports two execution modes:
- MOCK_HARNESS: Pure in-memory generator (deterministic, zero socket overhead).
- SITL_UDP: Full-duplex MAVLink 2.0 communication over real OS UDP sockets.

SCOPE CAVEAT:
verified: correct MAVLink 2.0 protocol implementation over real UDP socket;
not yet verified: compatibility with actual PX4 Autopilot firmware.
"""

from typing import Dict, Any, Generator, Tuple, Optional
import time
import numpy as np
import logging

from drone_abstraction.protocol import DroneMissionContract, MissionWaypoint, MAVCmdType
from drone_abstraction.mavlink_client import MAVLinkDroneClient
from drone_abstraction.sitl_server import PX4SITLEmulator
from data_pipeline.schema import TelemetryFrame, GeoPoint, DisasterClass
from data_pipeline.synthetic_generator import SyntheticAerialGenerator
from data_pipeline.india_augmentation import IndiaDomainAugmentor

logger = logging.getLogger("drone_abstraction.simulator")


class DroneSimulatorHarness:
    """Simulation harness supporting both in-memory mock and real UDP MAVLink 2.0 execution."""

    def __init__(
        self,
        synthetic_disaster: DisasterClass = DisasterClass.FLOOD,
        comm_mode: str = "MOCK_HARNESS",
        udp_port: int = 14550
    ):
        self.synthetic_disaster = synthetic_disaster
        self.comm_mode = comm_mode.upper()
        self.udp_port = udp_port

        self.generator = SyntheticAerialGenerator(image_size=(512, 512))
        self.battery_pct: float = 100.0
        self.current_altitude_m: float = 0.0
        self.is_armed: bool = False
        self.current_wp_idx: int = 0

    def execute_mission_stream(
        self, 
        mission: DroneMissionContract, 
        time_step_sec: float = 0.5
    ) -> Generator[Tuple[TelemetryFrame, np.ndarray, Dict[str, Any]], None, None]:
        """Generator executing mission and yielding telemetry and video frames."""
        if self.comm_mode == "SITL_UDP":
            yield from self._execute_sitl_udp_stream(mission, time_step_sec)
        else:
            yield from self._execute_mock_stream(mission, time_step_sec)

    def _execute_mock_stream(
        self,
        mission: DroneMissionContract,
        time_step_sec: float = 0.5
    ) -> Generator[Tuple[TelemetryFrame, np.ndarray, Dict[str, Any]], None, None]:
        """Deterministic, in-memory flight loop without network sockets."""
        self.is_armed = True
        self.battery_pct = 100.0

        for idx, wp in enumerate(mission.waypoints):
            self.current_wp_idx = idx
            self.current_altitude_m = wp.altitude_agl_m
            self.battery_pct = max(0.0, self.battery_pct - 2.5)

            # Generate simulated camera frame for this aerial vantage point
            img, _ = self.generator.generate_scene(
                disaster_type=self.synthetic_disaster,
                seed=42 + idx
            )
            if self.synthetic_disaster == DisasterClass.FLOOD:
                img = IndiaDomainAugmentor.augment_for_india(img, is_monsoon=True)

            telemetry = TelemetryFrame(
                frame_id=f"frame_sim_{mission.mission_id}_wp{idx}",
                timestamp=time.time(),
                gps=GeoPoint(
                    latitude=wp.latitude,
                    longitude=wp.longitude,
                    altitude_m=wp.altitude_agl_m
                ),
                altitude_agl_m=wp.altitude_agl_m,
                heading_deg=float((idx * 60) % 360)
            )

            status = {
                "mission_id": mission.mission_id,
                "current_waypoint": idx,
                "total_waypoints": len(mission.waypoints),
                "command": wp.command.value,
                "battery_pct": round(self.battery_pct, 1),
                "is_armed": self.is_armed,
                "in_failsafe": self.battery_pct <= mission.rth_battery_threshold_pct
            }

            yield telemetry, img, status

            if self.battery_pct <= mission.rth_battery_threshold_pct:
                logger.warning(f"[SimHarness] Low battery trigger ({self.battery_pct}%). Aborting to RTH.")
                break

        self.is_armed = False

    def _execute_sitl_udp_stream(
        self,
        mission: DroneMissionContract,
        time_step_sec: float = 0.5
    ) -> Generator[Tuple[TelemetryFrame, np.ndarray, Dict[str, Any]], None, None]:
        """Executes flight over genuine UDP socket using MAVLink 2.0 protocol."""
        server = PX4SITLEmulator(target_host="127.0.0.1", target_port=self.udp_port)
        client = MAVLinkDroneClient(
            connection_str=f"udpin:127.0.0.1:{self.udp_port}",
            heartbeat_timeout_sec=3.0,
            mission_timeout_sec=2.0
        )

        try:
            server.start()
            connected = client.connect(timeout=3.0)
            if not connected:
                raise ConnectionError(f"Failed to connect to SITL MAVLink socket on port {self.udp_port}")

            # Upload mission via MAVLink
            uploaded = client.upload_mission(mission)
            if not uploaded:
                raise RuntimeError(f"MAVLink mission upload rejected for {mission.mission_id}")

            # Arm vehicle and start mission
            client.arm_vehicle(True)
            client.start_mission()
            self.is_armed = True

            for idx, wp in enumerate(mission.waypoints):
                self.current_wp_idx = idx

                # Poll telemetry from MAVLink
                telem_res = client.poll_telemetry(timeout=0.6)
                if telem_res:
                    telemetry, status = telem_res
                else:
                    telemetry = TelemetryFrame(
                        frame_id=f"frame_sitl_{mission.mission_id}_wp{idx}",
                        timestamp=time.time(),
                        gps=GeoPoint(latitude=wp.latitude, longitude=wp.longitude, altitude_m=wp.altitude_agl_m),
                        altitude_agl_m=wp.altitude_agl_m,
                        heading_deg=float((idx * 60) % 360)
                    )
                    status = {
                        "current_waypoint": idx,
                        "battery_pct": round(client.latest_battery_pct, 1),
                        "is_armed": client.latest_armed,
                        "in_failsafe": False
                    }

                status["mission_id"] = mission.mission_id
                status["current_waypoint"] = idx
                status["autopilot_waypoint"] = client.latest_mission_seq
                status["total_waypoints"] = len(mission.waypoints)
                status["command"] = wp.command.value

                # Synthesize aerial visual frame for this vantage point
                img, _ = self.generator.generate_scene(disaster_type=self.synthetic_disaster, seed=42 + idx)
                if self.synthetic_disaster == DisasterClass.FLOOD:
                    img = IndiaDomainAugmentor.augment_for_india(img, is_monsoon=True)

                yield telemetry, img, status

                if status.get("in_failsafe", False) or status.get("battery_pct", 100.0) <= mission.rth_battery_threshold_pct:
                    client.return_to_launch()
                    break

            client.arm_vehicle(False)
            self.is_armed = False

        finally:
            client.close()
            server.stop()

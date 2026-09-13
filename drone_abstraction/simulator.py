"""PX4 SITL-Compatible Autonomous Drone Flight Simulator.
Executes DroneMissionContract, emits MAVLink-style telemetry, simulates battery draw,
and synthesizes live aerial video frames at each waypoint for on-scene computer vision.
"""

from typing import Dict, Any, Generator, Tuple, Optional
import time
import numpy as np
from drone_abstraction.protocol import DroneMissionContract, MissionWaypoint, MAVCmdType
from data_pipeline.schema import TelemetryFrame, GeoPoint, DisasterClass
from data_pipeline.synthetic_generator import SyntheticAerialGenerator
from data_pipeline.india_augmentation import IndiaDomainAugmentor


class DroneSimulatorHarness:
    """Simulation harness allowing 100% hardware-free mission and computer vision testing."""

    def __init__(self, synthetic_disaster: DisasterClass = DisasterClass.FLOOD):
        self.synthetic_disaster = synthetic_disaster
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
        """Generator simulating flight along waypoints, yielding telemetry and video frames."""
        self.is_armed = True
        self.battery_pct = 100.0

        for idx, wp in enumerate(mission.waypoints):
            self.current_wp_idx = idx
            
            # Simulate climb / descent
            self.current_altitude_m = wp.altitude_agl_m
            # Battery consumption: ~2% per waypoint leg
            self.battery_pct = max(0.0, self.battery_pct - 2.5)

            # Generate simulated camera frame for this aerial vantage point
            img, _ = self.generator.generate_scene(
                disaster_type=self.synthetic_disaster,
                seed=42 + idx
            )
            # Add realistic Indian monsoon/cyclone visual effects
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
                print(f"[SimHarness] Low battery trigger ({self.battery_pct}%). Aborting to RTH.")
                break

        self.is_armed = False

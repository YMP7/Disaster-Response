"""Hardware-Agnostic Drone Mission Protocol Contract.
Serializes high-level tactical survey intents into protocol-agnostic mission items
compatible with MAVLink (v1/v2), MAVSDK, ROS2, and PX4 SITL.
Zero business logic changes are required when switching from simulation to physical autopilots.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from data_pipeline.schema import GeoPoint, SensorModality


class MAVCmdType(str, Enum):
    TAKEOFF = "MAV_CMD_NAV_TAKEOFF"
    WAYPOINT = "MAV_CMD_NAV_WAYPOINT"
    LOITER_UNLIM = "MAV_CMD_NAV_LOITER_UNLIM"
    RETURN_TO_LAUNCH = "MAV_CMD_NAV_RETURN_TO_LAUNCH"
    SET_CAM_MODE = "MAV_CMD_SET_CAMERA_MODE"
    IMAGE_START_CAPTURE = "MAV_CMD_IMAGE_START_CAPTURE"
    IMAGE_STOP_CAPTURE = "MAV_CMD_IMAGE_STOP_CAPTURE"


# Numerical MAVLink Command IDs per MAVLink v2.0 Common Spec
MAV_CMD_NUMERICAL_MAP = {
    MAVCmdType.WAYPOINT: 16,               # MAV_CMD_NAV_WAYPOINT
    MAVCmdType.LOITER_UNLIM: 17,           # MAV_CMD_NAV_LOITER_UNLIM
    MAVCmdType.RETURN_TO_LAUNCH: 20,       # MAV_CMD_NAV_RETURN_TO_LAUNCH
    MAVCmdType.TAKEOFF: 22,                # MAV_CMD_NAV_TAKEOFF
    MAVCmdType.SET_CAM_MODE: 530,          # MAV_CMD_SET_CAMERA_MODE
    MAVCmdType.IMAGE_START_CAPTURE: 2000,  # MAV_CMD_IMAGE_START_CAPTURE
    MAVCmdType.IMAGE_STOP_CAPTURE: 2001,   # MAV_CMD_IMAGE_STOP_CAPTURE
}


class MissionWaypoint(BaseModel):
    seq: int
    command: MAVCmdType
    latitude: float
    longitude: float
    altitude_agl_m: float = Field(ge=0.0)  # Checked against 120m ceiling by DGCAComplianceChecker
    speed_mps: float = 8.0
    hold_time_sec: float = 0.0
    sensor_capture: bool = True

    @property
    def command_id(self) -> int:
        """Returns standard MAVLink uint16 numerical command ID."""
        return MAV_CMD_NUMERICAL_MAP.get(self.command, 16)


class DroneMissionContract(BaseModel):
    """Universal mission specification emitted by the AI dispatch engine."""
    mission_id: str
    target_zone_id: str
    pilot_in_command_id: str = "RPIC-IN-DGCA-CERTIFIED-09"
    sensor: SensorModality = SensorModality.OPTICAL_RGB
    cruise_altitude_agl_m: float = 60.0
    speed_mps: float = 10.0
    rth_battery_threshold_pct: float = 25.0
    waypoints: List[MissionWaypoint]
    failsafe_rules: Dict[str, str] = Field(default_factory=lambda: {
        "gps_loss": "land_immediately",
        "link_loss": "return_to_home",
        "battery_critical": "return_to_home",
        "geofence_breach": "loiter_and_alert_rpic"
    })

    def to_mavlink_mission_items(self) -> List[Dict[str, Any]]:
        """Exports to standard MAVLink MISSION_ITEM_INT dictionary structure."""
        items = []
        for wp in self.waypoints:
            items.append({
                "seq": wp.seq,
                "frame": 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
                "command": wp.command.value,
                "command_id": wp.command_id,
                "current": 1 if wp.seq == 0 else 0,
                "autocontinue": 1,
                "param1": wp.hold_time_sec,
                "param2": 2.0,  # Accept radius meters
                "param3": 0.0,
                "param4": 0.0,
                "x": int(round(wp.latitude * 1e7)),
                "y": int(round(wp.longitude * 1e7)),
                "z": float(wp.altitude_agl_m)
            })
        return items

    @classmethod
    def create_grid_survey_mission(
        cls,
        mission_id: str,
        zone_id: str,
        center_gps: GeoPoint,
        grid_radius_m: float = 500.0,
        altitude_agl_m: float = 60.0
    ) -> "DroneMissionContract":
        """Generates an automated lawnmower survey pattern around an incident epicenter."""
        import math
        # 1 deg lat approx 111,000m
        delta_lat = (grid_radius_m / 111000.0)
        delta_lon = (grid_radius_m / (111000.0 * math.cos(math.radians(center_gps.latitude))))

        wps = [
            MissionWaypoint(
                seq=0, command=MAVCmdType.TAKEOFF,
                latitude=center_gps.latitude, longitude=center_gps.longitude,
                altitude_agl_m=altitude_agl_m
            ),
            MissionWaypoint(
                seq=1, command=MAVCmdType.WAYPOINT,
                latitude=center_gps.latitude - delta_lat, longitude=center_gps.longitude - delta_lon,
                altitude_agl_m=altitude_agl_m
            ),
            MissionWaypoint(
                seq=2, command=MAVCmdType.WAYPOINT,
                latitude=center_gps.latitude + delta_lat, longitude=center_gps.longitude - delta_lon,
                altitude_agl_m=altitude_agl_m
            ),
            MissionWaypoint(
                seq=3, command=MAVCmdType.WAYPOINT,
                latitude=center_gps.latitude + delta_lat, longitude=center_gps.longitude + delta_lon,
                altitude_agl_m=altitude_agl_m
            ),
            MissionWaypoint(
                seq=4, command=MAVCmdType.WAYPOINT,
                latitude=center_gps.latitude - delta_lat, longitude=center_gps.longitude + delta_lon,
                altitude_agl_m=altitude_agl_m
            ),
            MissionWaypoint(
                seq=5, command=MAVCmdType.RETURN_TO_LAUNCH,
                latitude=center_gps.latitude, longitude=center_gps.longitude,
                altitude_agl_m=altitude_agl_m
            )
        ]

        return cls(
            mission_id=mission_id,
            target_zone_id=zone_id,
            cruise_altitude_agl_m=altitude_agl_m,
            waypoints=wps
        )

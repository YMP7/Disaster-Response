"""Production-Grade MAVLink v2.0 UDP Client for Autonomous Drone Operations.
Connects via pymavlink over real OS UDP sockets (e.g. GCS port 14550 or companion port 14540).
Executes standard MAVLink Mission Protocol, command long sequences (arm/disarm/takeoff/RTL),
and streams binary telemetry frames.

SCOPE CAVEAT:
verified: correct MAVLink 2.0 protocol implementation over real UDP socket;
not yet verified: compatibility with actual PX4 Autopilot firmware.
"""

import time
import math
import socket
import logging
from typing import Optional, Dict, Any, List, Tuple
from pymavlink import mavutil

from data_pipeline.schema import TelemetryFrame, GeoPoint
from drone_abstraction.protocol import DroneMissionContract, MissionWaypoint

logger = logging.getLogger("drone_abstraction.mavlink_client")


class MAVLinkDroneClient:
    """Client for bidirectional MAVLink 2.0 communication over UDP sockets."""

    def __init__(
        self,
        connection_str: str = "udpin:0.0.0.0:14550",
        source_system: int = 255,      # Standard GCS / Companion Computer SysID
        source_component: int = 190,   # MAV_COMP_ID_MISSIONPLANNER / Companion
        target_system: int = 1,        # Default Autopilot SysID
        target_component: int = 1,      # Default Autopilot CompID
        heartbeat_timeout_sec: float = 5.0,
        mission_timeout_sec: float = 3.0,
        max_retries: int = 3
    ):
        self.connection_str = connection_str
        self.source_system = source_system
        self.source_component = source_component
        self.target_system = target_system
        self.target_component = target_component
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.mission_timeout_sec = mission_timeout_sec
        self.max_retries = max_retries

        self.master: Optional[mavutil.mavlink_connection] = None
        self.last_heartbeat_time: float = 0.0
        self.is_connected: bool = False
        self.autopilot_type: Optional[int] = None
        self.vehicle_type: Optional[int] = None
        self.base_mode: int = 0
        self.custom_mode: int = 0
        self.system_status: int = 0

        # Cached telemetry state
        self.latest_lat: float = 0.0
        self.latest_lon: float = 0.0
        self.latest_alt_agl_m: float = 0.0
        self.latest_heading_deg: float = 0.0
        self.latest_battery_pct: float = 100.0
        self.latest_armed: bool = False
        self.latest_mission_seq: int = 0

    def connect(self, timeout: float = 5.0) -> bool:
        """Binds/connects to the UDP socket and waits for initial heartbeat."""
        logger.info(f"Connecting to MAVLink endpoint: {self.connection_str}")
        self.master = mavutil.mavlink_connection(
            self.connection_str,
            source_system=self.source_system,
            source_component=self.source_component
        )
        if hasattr(socket, 'SIO_UDP_CONNRESET') and hasattr(self.master, 'port') and hasattr(self.master.port, 'ioctl'):
            try:
                self.master.port.ioctl(socket.SIO_UDP_CONNRESET, False)
            except Exception:
                pass
        return self.wait_for_heartbeat(timeout=timeout)

    def send_heartbeat(self):
        """Sends GCS/Companion heartbeat to maintain link."""
        if not self.master:
            return
        try:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,  # base_mode
                0,  # custom_mode
                0   # system_status
            )
        except (ConnectionResetError, OSError):
            pass

    def wait_for_heartbeat(self, timeout: Optional[float] = None) -> bool:
        """Blocks until a HEARTBEAT message is received from the target autopilot."""
        if not self.master:
            return False
        timeout = timeout or self.heartbeat_timeout_sec
        start_t = time.time()

        while time.time() - start_t < timeout:
            try:
                msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=min(0.5, timeout))
                if msg:
                    self.target_system = msg.get_srcSystem()
                    self.target_component = msg.get_srcComponent()
                    self.autopilot_type = msg.autopilot
                    self.vehicle_type = msg.type
                    self.base_mode = msg.base_mode
                    self.custom_mode = msg.custom_mode
                    self.system_status = msg.system_status
                    self.latest_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    self.last_heartbeat_time = time.time()
                    self.is_connected = True
                    logger.info(f"Heartbeat received from sysid={self.target_system}, compid={self.target_component}, autopilot={self.autopilot_type}")
                    return True
            except (ConnectionResetError, OSError):
                time.sleep(0.05)

        logger.warning(f"No heartbeat received within {timeout}s from {self.connection_str}")
        return False

    def is_heartbeat_healthy(self) -> bool:
        """Audits link freshness against timeout."""
        if not self.is_connected or self.last_heartbeat_time == 0.0:
            return False
        return (time.time() - self.last_heartbeat_time) <= self.heartbeat_timeout_sec

    def upload_mission(self, mission: DroneMissionContract) -> bool:
        """Uploads mission waypoints using the standard MAVLink Mission Protocol.
        Implements MISSION_COUNT -> MISSION_REQUEST_INT -> MISSION_ITEM_INT -> MISSION_ACK.
        Includes retry and lossy-network tolerance.
        """
        if not self.master:
            raise ConnectionError("MAVLink client not connected")

        items = mission.to_mavlink_mission_items()
        count = len(items)
        if count == 0:
            return False

        logger.info(f"Starting mission upload: {count} items for mission {mission.mission_id}")

        # Step 1: Send MISSION_COUNT with retry
        ack_received = False
        for attempt in range(self.max_retries):
            # Clear input buffer
            while self.master.recv_match(blocking=False):
                pass

            self.master.mav.mission_count_send(
                self.target_system,
                self.target_component,
                count,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            )

            # Step 2: Receive requests for each sequence number
            upload_success = True
            uploaded_seqs = set()

            start_t = time.time()
            while len(uploaded_seqs) < count:
                if (time.time() - start_t) > (self.mission_timeout_sec * 2):
                    upload_success = False
                    break

                req = self.master.recv_match(
                    type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'],
                    blocking=True,
                    timeout=self.mission_timeout_sec
                )

                if not req:
                    # Packet drop / timeout: retry sending items or break to outer attempt
                    upload_success = False
                    break

                seq = req.seq
                if 0 <= seq < count:
                    item = items[seq]
                    self.master.mav.mission_item_int_send(
                        self.target_system,
                        self.target_component,
                        item["seq"],
                        item["frame"],
                        item["command_id"],
                        item["current"],
                        item["autocontinue"],
                        float(item["param1"]),
                        float(item["param2"]),
                        float(item["param3"]),
                        float(item["param4"]),
                        int(item["x"]),
                        int(item["y"]),
                        float(item["z"]),
                        mavutil.mavlink.MAV_MISSION_TYPE_MISSION
                    )
                    uploaded_seqs.add(seq)

            if not upload_success:
                logger.warning(f"Mission upload sequence stalled on attempt {attempt+1}/{self.max_retries}. Retrying...")
                continue

            # Step 3: Wait for MISSION_ACK
            ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=self.mission_timeout_sec)
            if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                logger.info(f"Mission {mission.mission_id} uploaded and ACCEPTED by autopilot")
                return True
            elif ack:
                logger.error(f"Mission rejected with ACK type: {ack.type}")
                return False

        logger.error(f"Mission upload failed after {self.max_retries} attempts")
        return False

    def arm_vehicle(self, arm: bool = True, timeout: float = 3.0) -> bool:
        """Arms or disarms the flight controller via MAV_CMD_COMPONENT_ARM_DISARM."""
        if not self.master:
            raise ConnectionError("MAVLink client not connected")

        param1 = 1.0 if arm else 0.0
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,       # confirmation
            param1,  # 1 = arm, 0 = disarm
            0, 0, 0, 0, 0, 0
        )

        try:
            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=timeout)
        except (ConnectionResetError, OSError):
            ack = None
        if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.latest_armed = arm
                logger.info(f"Vehicle {'ARMED' if arm else 'DISARMED'} successfully")
                return True
        return False

    def start_mission(self, timeout: float = 3.0) -> bool:
        """Commands autopilot to begin executing the active mission."""
        if not self.master:
            raise ConnectionError("MAVLink client not connected")

        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0,
            0, 0, 0, 0, 0, 0, 0
        )

        try:
            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=timeout)
        except (ConnectionResetError, OSError):
            ack = None
        if ack and ack.command == mavutil.mavlink.MAV_CMD_MISSION_START:
            return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
        return False

    def return_to_launch(self, timeout: float = 3.0) -> bool:
        """Dispatches emergency or mission-complete Return-To-Launch command."""
        if not self.master:
            raise ConnectionError("MAVLink client not connected")

        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0,
            0, 0, 0, 0, 0, 0, 0
        )

        try:
            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=timeout)
        except (ConnectionResetError, OSError):
            ack = None
        if ack and ack.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
        return False

    def poll_telemetry(self, timeout: float = 0.5) -> Optional[Tuple[TelemetryFrame, Dict[str, Any]]]:
        """Reads incoming MAVLink frames and updates state.
        Returns a (TelemetryFrame, status_dict) pair if new telemetry arrived.
        """
        if not self.master:
            return None

        # Drain and process all pending messages
        updated = False
        start_t = time.time()
        while True:
            try:
                msg = self.master.recv_match(blocking=False)
            except (ConnectionResetError, OSError):
                msg = None
            if not msg:
                if not updated and (time.time() - start_t) < timeout:
                    time.sleep(0.01)
                    continue
                break

            msg_type = msg.get_type()

            if msg_type == 'HEARTBEAT':
                self.last_heartbeat_time = time.time()
                self.latest_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.system_status = msg.system_status
                updated = True

            elif msg_type == 'GLOBAL_POSITION_INT':
                self.latest_lat = msg.lat / 1e7
                self.latest_lon = msg.lon / 1e7
                self.latest_alt_agl_m = max(0.0, msg.relative_alt / 1000.0)
                self.latest_heading_deg = (msg.hdg / 100.0) if msg.hdg != 65535 else 0.0
                updated = True

            elif msg_type == 'SYS_STATUS':
                if msg.battery_remaining != -1:
                    self.latest_battery_pct = float(msg.battery_remaining)
                updated = True

            elif msg_type == 'MISSION_CURRENT':
                self.latest_mission_seq = msg.seq
                updated = True

        if updated:
            frame = TelemetryFrame(
                frame_id=f"frame_mavlink_seq{self.latest_mission_seq}_{int(time.time()*1000)}",
                timestamp=time.time(),
                gps=GeoPoint(
                    latitude=self.latest_lat,
                    longitude=self.latest_lon,
                    altitude_m=self.latest_alt_agl_m
                ),
                altitude_agl_m=self.latest_alt_agl_m,
                heading_deg=self.latest_heading_deg
            )
            status = {
                "current_waypoint": self.latest_mission_seq,
                "battery_pct": round(self.latest_battery_pct, 1),
                "is_armed": self.latest_armed,
                "is_link_healthy": self.is_heartbeat_healthy(),
                "in_failsafe": not self.is_heartbeat_healthy() or self.latest_battery_pct < 25.0
            }
            return frame, status

        return None

    def close(self):
        """Cleanly releases socket resources."""
        if self.master:
            try:
                self.master.close()
            except Exception:
                pass
            self.master = None
            self.is_connected = False
            logger.info("MAVLink UDP socket closed")

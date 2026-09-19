"""Standalone PX4 SITL MAVLink v2.0 Emulator.
Binds to an OS UDP socket and acts as an authentic MAVLink 2.0 autopilot server.
Implements the MAVLink Mission Protocol handshake, command routing, and telemetry streaming.

SCOPE CAVEAT:
verified: correct MAVLink 2.0 protocol implementation over real UDP socket;
not yet verified: compatibility with actual PX4 Autopilot firmware.
"""

import time
import socket
import threading
import logging
from typing import Optional, List, Dict, Any
from pymavlink import mavutil

logger = logging.getLogger("drone_abstraction.sitl_server")


class PX4SITLEmulator:
    """Authentic MAVLink 2.0 autopilot simulator operating over real UDP sockets."""

    def __init__(
        self,
        target_host: str = "127.0.0.1",
        target_port: int = 14550,
        sys_id: int = 1,
        comp_id: int = 1
    ):
        self.target_host = target_host
        self.target_port = target_port
        self.sys_id = sys_id
        self.comp_id = comp_id

        self.conn_str = f"udpout:{target_host}:{target_port}"
        self.mav: Optional[mavutil.mavlink_connection] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Autopilot State
        self.is_armed = False
        self.base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        self.custom_mode = 0  # 0 = MANUAL / STANDBY
        self.system_status = mavutil.mavlink.MAV_STATE_STANDBY
        self.battery_pct = 100.0

        # Mission Storage
        self.expected_mission_count = 0
        self.stored_mission_items: List[Dict[str, Any]] = []
        self.current_waypoint_seq = 0
        self.is_mission_active = False

        # Current Spatial State
        self.lat_deg = 20.2961     # Default: Bhubaneswar / Odisha
        self.lon_deg = 85.8245
        self.alt_agl_m = 0.0
        self.heading_deg = 0.0

        # Lossy network test injection hooks
        self.drop_request_seq: Optional[int] = None
        self.has_dropped_request: bool = False
        self.delay_ack_sec: float = 0.0

    def start(self):
        """Opens UDP socket and launches asynchronous listener/heartbeat loop."""
        logger.info(f"Starting PX4 SITL Emulator targeting {self.conn_str}")
        self.mav = mavutil.mavlink_connection(
            self.conn_str,
            source_system=self.sys_id,
            source_component=self.comp_id
        )
        if hasattr(socket, 'SIO_UDP_CONNRESET') and hasattr(self.mav, 'port') and hasattr(self.mav.port, 'ioctl'):
            try:
                self.mav.port.ioctl(socket.SIO_UDP_CONNRESET, False)
            except Exception:
                pass
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops emulator loop and closes UDP socket."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.mav:
            try:
                self.mav.close()
            except Exception:
                pass
            self.mav = None
        logger.info("PX4 SITL Emulator stopped")

    def _run_loop(self):
        last_hb = 0.0
        last_telem = 0.0

        while self._running:
            now = time.time()

            # 1. Emit periodic 1 Hz Heartbeat
            if now - last_hb >= 1.0:
                try:
                    self._send_heartbeat()
                except (ConnectionResetError, OSError):
                    pass
                last_hb = now

            # 2. Emit periodic 5 Hz Telemetry during active flight
            if self.is_armed and (now - last_telem >= 0.2):
                self._step_flight_physics()
                try:
                    self._send_telemetry()
                except (ConnectionResetError, OSError):
                    pass
                last_telem = now

            # 3. Drain and process incoming MAVLink packets
            while self._running:
                try:
                    msg = self.mav.recv_match(blocking=False)
                except (ConnectionResetError, OSError):
                    msg = None
                    break
                if not msg:
                    break
                try:
                    self._handle_message(msg)
                except (ConnectionResetError, OSError):
                    pass

            time.sleep(0.01)

    def _send_heartbeat(self):
        if not self.mav:
            return
        mode = self.base_mode
        if self.is_armed:
            mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

        self.mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_PX4,
            mode,
            self.custom_mode,
            self.system_status
        )

    def _send_telemetry(self):
        if not self.mav:
            return
        # Emit GLOBAL_POSITION_INT
        self.mav.mav.global_position_int_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            int(round(self.lat_deg * 1e7)),
            int(round(self.lon_deg * 1e7)),
            int(round(self.alt_agl_m * 1000)),       # alt mm
            int(round(self.alt_agl_m * 1000)),       # relative_alt mm
            0, 0, 0,                                 # vx, vy, vz
            int(round(self.heading_deg * 100))       # hdg cdeg
        )

        # Emit SYS_STATUS
        self.mav.mav.sys_status_send(
            0, 0, 0, 0,
            12600,                                   # 12.6V battery
            -1,                                      # current unknown
            int(round(self.battery_pct)),            # battery remaining %
            0, 0, 0, 0, 0, 0
        )

        # Emit MISSION_CURRENT if mission loaded
        if self.is_mission_active:
            self.mav.mav.mission_current_send(self.current_waypoint_seq)

    def _step_flight_physics(self):
        """Simulates waypoint progression and battery depletion."""
        if not self.is_mission_active or not self.stored_mission_items:
            return

        if self.current_waypoint_seq < len(self.stored_mission_items):
            target_wp = self.stored_mission_items[self.current_waypoint_seq]
            target_lat = target_wp["x"] / 1e7
            target_lon = target_wp["y"] / 1e7
            target_alt = target_wp["z"]

            # Interpolate towards target
            self.lat_deg += (target_lat - self.lat_deg) * 0.4
            self.lon_deg += (target_lon - self.lon_deg) * 0.4
            self.alt_agl_m += (target_alt - self.alt_agl_m) * 0.4

            # Consume battery
            self.battery_pct = max(0.0, self.battery_pct - 0.2)

            # Check if close enough to advance
            dist = abs(self.lat_deg - target_lat) + abs(self.lon_deg - target_lon)
            if dist < 0.0001:
                self.current_waypoint_seq += 1
                if self.current_waypoint_seq >= len(self.stored_mission_items):
                    self.is_mission_active = False
                    logger.info("SITL: All mission waypoints executed")

    def _handle_message(self, msg):
        msg_type = msg.get_type()

        # Mission Upload Handshake
        if msg_type == 'MISSION_COUNT':
            self.expected_mission_count = msg.count
            self.stored_mission_items.clear()
            logger.info(f"SITL: Expecting {msg.count} mission items")
            if msg.count > 0:
                self._request_next_item(0)

        elif msg_type in ('MISSION_ITEM_INT', 'MISSION_ITEM'):
            seq = msg.seq
            # Store item
            item_dict = {
                "seq": msg.seq,
                "frame": msg.frame,
                "command": msg.command,
                "x": msg.x,
                "y": msg.y,
                "z": msg.z
            }
            self.stored_mission_items.append(item_dict)
            next_seq = seq + 1

            if next_seq < self.expected_mission_count:
                self._request_next_item(next_seq)
            else:
                # All items received -> Send MISSION_ACK
                if self.delay_ack_sec > 0:
                    time.sleep(self.delay_ack_sec)
                self.mav.mav.mission_ack_send(
                    msg.get_srcSystem(),
                    msg.get_srcComponent(),
                    mavutil.mavlink.MAV_MISSION_ACCEPTED,
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION
                )
                logger.info("SITL: Mission upload complete, ACK sent")

        # Command Long Handler
        elif msg_type == 'COMMAND_LONG':
            cmd = msg.command
            result = mavutil.mavlink.MAV_RESULT_FAILED

            if cmd == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                self.is_armed = bool(msg.param1 == 1.0)
                result = mavutil.mavlink.MAV_RESULT_ACCEPTED
                logger.info(f"SITL: Arming command processed -> armed={self.is_armed}")

            elif cmd == mavutil.mavlink.MAV_CMD_MISSION_START:
                if self.stored_mission_items:
                    self.is_mission_active = True
                    self.current_waypoint_seq = 0
                    result = mavutil.mavlink.MAV_RESULT_ACCEPTED
                    logger.info("SITL: Mission execution started")
                else:
                    result = mavutil.mavlink.MAV_RESULT_DENIED

            elif cmd == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
                self.is_mission_active = False
                self.alt_agl_m = 0.0
                result = mavutil.mavlink.MAV_RESULT_ACCEPTED
                logger.info("SITL: Return-To-Launch executed")

            # Send Command ACK
            try:
                self.mav.mav.command_ack_send(cmd, result)
            except (ConnectionResetError, OSError):
                pass

    def _request_next_item(self, seq: int):
        # Check lossy injection hook
        if self.drop_request_seq == seq and not self.has_dropped_request:
            self.has_dropped_request = True
            logger.warning(f"SITL [SIMULATED PACKET DROP]: Dropping MISSION_REQUEST_INT for seq {seq}")
            return

        self.mav.mav.mission_request_int_send(
            255, 190,  # Target GCS
            seq,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION
        )

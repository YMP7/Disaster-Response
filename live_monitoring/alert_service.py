"""Live Public Disaster Feed Monitoring and Alert Dispatch Service.
Monitors SACHET / CAP feeds from IMD, CWC, INCOIS, and NRSC.
Geofences to operational radius, deduplicates, and emits typed internal alert events.
"""

from typing import List, Optional, Callable, Dict, Any
from pydantic import BaseModel, Field
import time
from data_pipeline.schema import DisasterClass, GeoPoint
from live_monitoring.cap_parser import SACHETCAPParser, CAPAlert
from live_monitoring.geofence import GeofenceEngine, MonitoredZone


class TypedAlertEvent(BaseModel):
    event_id: str
    source: str
    timestamp: float
    disaster_class: DisasterClass
    severity: str
    headline: str
    affected_zone_id: str
    affected_zone_name: str
    epicenter_gps: GeoPoint
    radius_km: float
    is_escalating: bool
    requires_drone_recon: bool
    raw_cap: Optional[Dict[str, Any]] = None


class LiveMonitoringService:
    """Continuous polling and event dispatch service."""

    def __init__(self, geofence_engine: Optional[GeofenceEngine] = None):
        self.geofence = geofence_engine or GeofenceEngine()
        self.active_subscribers: List[Callable[[TypedAlertEvent], None]] = []

    def subscribe(self, callback: Callable[[TypedAlertEvent], None]):
        self.active_subscribers.append(callback)

    def process_incoming_cap(self, cap_xml_or_json: str) -> Optional[TypedAlertEvent]:
        """Ingests a CAP message, tests geofence, and dispatches if relevant."""
        alert = SACHETCAPParser.parse_xml(cap_xml_or_json)
        is_relevant, is_escalation, zone, reason = self.geofence.evaluate_and_deduplicate(alert)

        if not is_relevant or zone is None:
            return None

        circle_parts = (alert.info.area.circle or f"{zone.latitude},{zone.longitude} 25.0").split()
        lat, lon = [float(c) for c in circle_parts[0].split(",")]
        radius = float(circle_parts[1]) if len(circle_parts) > 1 else 25.0

        event = TypedAlertEvent(
            event_id=alert.identifier,
            source=alert.sender,
            timestamp=time.time(),
            disaster_class=alert.info.disaster_class_mapped,
            severity=alert.info.severity,
            headline=alert.info.headline,
            affected_zone_id=zone.zone_id,
            affected_zone_name=zone.name,
            epicenter_gps=GeoPoint(latitude=lat, longitude=lon),
            radius_km=radius,
            is_escalating=is_escalation,
            requires_drone_recon=(alert.info.severity in ["Severe", "Extreme"]),
            raw_cap=alert.model_dump()
        )

        for sub in self.active_subscribers:
            try:
                sub(event)
            except Exception as e:
                print(f"[AlertService] Subscriber error: {e}")

        return event

    @staticmethod
    def get_sample_sachet_odisha_cyclone_alert() -> str:
        """Sample OASIS CAP XML representing an IMD cyclone warning for coastal Odisha."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>SACHET-IMD-2026-CYC-042</identifier>
  <sender>in-imd-cyclone-warning-centre</sender>
  <sent>2026-09-13T10:00:00+05:30</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <category>Met</category>
    <event>Severe Cyclonic Storm</event>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Observed</certainty>
    <headline>Severe Cyclonic Storm warning with storm surge and localized inundation</headline>
    <description>Severe Cyclonic storm crossing coastal corridor near Cuttack/Puri district. Heavy squally winds with intense localized inundation expected.</description>
    <instruction>Evacuate low lying areas. Suspend fishing and drone operations below commercial ceilings.</instruction>
    <area>
      <areaDesc>Cuttack and Puri coastal corridor, Odisha</areaDesc>
      <circle>20.4625,85.8828 45.0</circle>
    </area>
  </info>
</alert>"""

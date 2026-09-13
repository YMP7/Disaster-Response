"""Geofencing, Spatial Intersection, and Deduplication Engine.
Evaluates CAP alert circles and polygons against monitored operational centers
using spherical distance and geometric intersection.
"""

from typing import Tuple, List, Optional, Dict, Set
import math
from pydantic import BaseModel
from live_monitoring.cap_parser import CAPAlert


class MonitoredZone(BaseModel):
    zone_id: str
    name: str
    latitude: float
    longitude: float
    radius_km: float


class GeofenceEngine:
    """Evaluates spatial containment and manages alert deduplication."""

    def __init__(self, monitored_zones: Optional[List[MonitoredZone]] = None):
        self.monitored_zones: Dict[str, MonitoredZone] = {
            z.zone_id: z for z in (monitored_zones or [
                # Default monitored coastal hubs
                MonitoredZone(zone_id="odisha_cuttack", name="Cuttack-Bhubaneswar Hub", latitude=20.4625, longitude=85.8828, radius_km=80.0),
                MonitoredZone(zone_id="kerala_kochi", name="Ernakulam-Periyar Basin", latitude=9.9816, longitude=76.2999, radius_km=75.0),
                MonitoredZone(zone_id="uttarakhand_chamoli", name="Chamoli Valley", latitude=30.4075, longitude=79.3245, radius_km=60.0)
            ])
        }
        self.seen_alerts: Dict[str, str] = {}  # alert_id -> severity

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates great-circle distance between two points on the Earth (km)."""
        r = 6371.0  # Earth radius in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2.0) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda / 2.0) ** 2)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def intersects_monitored_zone(self, alert: CAPAlert) -> Tuple[bool, Optional[MonitoredZone], float]:
        """Checks whether the CAP alert circle/polygon intersects any monitored zone."""
        circle_str = alert.info.area.circle
        if circle_str:
            # Format: "lat,lon radius_km"
            parts = circle_str.strip().split()
            coords = parts[0].split(",")
            alert_lat, alert_lon = float(coords[0]), float(coords[1])
            alert_radius = float(parts[1]) if len(parts) > 1 else 10.0

            for zone in self.monitored_zones.values():
                dist = self.haversine_distance_km(zone.latitude, zone.longitude, alert_lat, alert_lon)
                if dist <= (zone.radius_km + alert_radius):
                    return True, zone, round(dist, 2)

        # Fallback: check text description heuristics if coordinates omitted
        for zone in self.monitored_zones.values():
            if zone.name.lower() in alert.info.area.area_desc.lower():
                return True, zone, 0.0

        return False, None, 0.0

    def evaluate_and_deduplicate(self, alert: CAPAlert) -> Tuple[bool, bool, Optional[MonitoredZone], str]:
        """Returns: (is_relevant, is_escalation, matched_zone, reason)"""
        intersects, zone, dist = self.intersects_monitored_zone(alert)
        if not intersects:
            return False, False, None, "Alert outside monitored operational radius"

        prev_severity = self.seen_alerts.get(alert.identifier)
        curr_severity = alert.info.severity

        if prev_severity == curr_severity:
            return False, False, zone, f"Duplicate alert {alert.identifier} with identical severity ({curr_severity})"

        is_escalation = False
        severity_rank = {"Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
        if prev_severity and severity_rank.get(curr_severity, 0) > severity_rank.get(prev_severity, 0):
            is_escalation = True

        self.seen_alerts[alert.identifier] = curr_severity
        return True, is_escalation, zone, f"Intersects {zone.name} at {dist}km (Severity: {curr_severity})"

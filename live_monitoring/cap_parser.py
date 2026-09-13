"""Oasis Common Alerting Protocol (CAP v1.2) Parser for India's SACHET Platform.
Parses standardized disaster warning feeds aggregating:
- India Meteorological Department (IMD) cyclones, severe weather
- Central Water Commission (CWC) riverine flood alerts
- Indian National Centre for Ocean Information Services (INCOIS) tsunami/surge
- National Remote Sensing Centre (NRSC) / Bhuvan inundation maps
"""

from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from data_pipeline.schema import DisasterClass, GeoPoint


class CAPArea(BaseModel):
    area_desc: str
    circle: Optional[str] = None  # Format: "lat,lon radius_km"
    polygon: Optional[str] = None  # Space-delimited "lat,lon" pairs


class CAPInfo(BaseModel):
    category: str
    event: str
    urgency: str       # Immediate, Expected, Future, Past, Unknown
    severity: str      # Extreme, Severe, Moderate, Minor, Unknown
    certainty: str     # Observed, Likely, Possible, Unlikely, Unknown
    event_code: Optional[str] = None
    headline: str
    description: str
    instruction: Optional[str] = None
    area: CAPArea
    disaster_class_mapped: DisasterClass


class CAPAlert(BaseModel):
    identifier: str
    sender: str        # e.g., "in-imd-hq", "in-cwc-flood", "in-incois-tsunami"
    sent: str
    status: str        # Actual, Exercise, System, Test, Draft
    msg_type: str      # Alert, Update, Cancel, Ack, Error
    scope: str         # Public, Restricted, Private
    info: CAPInfo


class SACHETCAPParser:
    """Parses XML and JSON CAP v1.2 documents from SACHET."""

    DISASTER_KEYWORDS = {
        "cyclone": DisasterClass.CYCLONE_STORM,
        "storm": DisasterClass.CYCLONE_STORM,
        "flood": DisasterClass.FLOOD,
        "inundation": DisasterClass.FLOOD,
        "heavy rain": DisasterClass.FLOOD,
        "earthquake": DisasterClass.EARTHQUAKE_COLLAPSE,
        "tremor": DisasterClass.EARTHQUAKE_COLLAPSE,
        "landslide": DisasterClass.LANDSLIDE,
        "debris flow": DisasterClass.LANDSLIDE,
        "fire": DisasterClass.WILDFIRE,
        "forest fire": DisasterClass.WILDFIRE,
        "chemical": DisasterClass.INDUSTRIAL_CHEMICAL,
        "gas leak": DisasterClass.INDUSTRIAL_CHEMICAL,
        "heatwave": DisasterClass.DROUGHT_HEATWAVE,
        "drought": DisasterClass.DROUGHT_HEATWAVE
    }

    @classmethod
    def map_event_to_class(cls, event_name: str, description: str) -> DisasterClass:
        text = f"{event_name} {description}".lower()
        for kw, dclass in cls.DISASTER_KEYWORDS.items():
            if kw in text:
                return dclass
        return DisasterClass.UNCERTAIN_NEEDS_REVIEW

    @classmethod
    def parse_xml(cls, xml_content: str) -> CAPAlert:
        root = ET.fromstring(xml_content)
        # Handle namespaces if present
        ns = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
        find = lambda tag: root.find(f".//{tag}") or root.find(f".//cap:{tag}", ns)

        identifier = getattr(find("identifier"), "text", "UNKNOWN_ID")
        sender = getattr(find("sender"), "text", "UNKNOWN_SENDER")
        sent = getattr(find("sent"), "text", "")
        status = getattr(find("status"), "text", "Actual")
        msg_type = getattr(find("msgType"), "text", "Alert")
        scope = getattr(find("scope"), "text", "Public")

        event = getattr(find("event"), "text", "General Disaster")
        severity = getattr(find("severity"), "text", "Moderate")
        urgency = getattr(find("urgency"), "text", "Immediate")
        certainty = getattr(find("certainty"), "text", "Observed")
        headline = getattr(find("headline"), "text", event)
        desc = getattr(find("description"), "text", "")
        area_desc = getattr(find("areaDesc"), "text", "Operational Region")
        circle = getattr(find("circle"), "text", None)
        polygon = getattr(find("polygon"), "text", None)

        dclass = cls.map_event_to_class(event, desc)

        return CAPAlert(
            identifier=identifier,
            sender=sender,
            sent=sent,
            status=status,
            msg_type=msg_type,
            scope=scope,
            info=CAPInfo(
                category=getattr(find("category"), "text", "Safety"),
                event=event,
                urgency=urgency,
                severity=severity,
                certainty=certainty,
                headline=headline,
                description=desc,
                area=CAPArea(area_desc=area_desc, circle=circle, polygon=polygon),
                disaster_class_mapped=dclass
            )
        )

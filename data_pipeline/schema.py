"""Unified Data Schemas for Disaster Ingestion, Vision Models, and Telemetry.
Standardizes representation across AIDER, xBD, FloodNet, RescueNet, and Indian sensor feeds.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import numpy as np


class DisasterClass(str, Enum):
    FLOOD = "flood"
    CYCLONE_STORM = "cyclone_storm"
    EARTHQUAKE_COLLAPSE = "earthquake_collapse"
    LANDSLIDE = "landslide"
    WILDFIRE = "wildfire"
    INDUSTRIAL_CHEMICAL = "industrial_chemical"
    DROUGHT_HEATWAVE = "drought_heatwave"
    NORMAL_SCENE = "normal_scene"
    UNCERTAIN_NEEDS_REVIEW = "uncertain_needs_review"


class DamageGrade(int, Enum):
    """Standardized 4-level ordinal scale (xBD format)."""
    NO_DAMAGE = 0
    MINOR_DAMAGE = 1
    MAJOR_DAMAGE = 2
    DESTROYED = 3


class RoadPassability(str, Enum):
    """Road status classification (RescueNet format)."""
    ROAD_CLEAR = "road_clear"
    ROAD_BLOCKED = "road_blocked"
    UNKNOWN = "unknown"


class SensorModality(str, Enum):
    OPTICAL_RGB = "optical_rgb"
    SAR_BACKSCATTER = "sar_backscatter"
    THERMAL_IR = "thermal_ir"
    MULTISPECTRAL = "multispectral"


class BoundingBox(BaseModel):
    """Normalized [ymin, xmin, ymax, xmax] coordinates (0.0 to 1.0)."""
    ymin: float = Field(ge=0.0, le=1.0)
    xmin: float = Field(ge=0.0, le=1.0)
    ymax: float = Field(ge=0.0, le=1.0)
    xmax: float = Field(ge=0.0, le=1.0)

    @property
    def area(self) -> float:
        return max(0.0, self.ymax - self.ymin) * max(0.0, self.xmax - self.xmin)


class BuildingFootprint(BaseModel):
    building_id: str
    bbox: BoundingBox
    damage_grade: DamageGrade
    confidence: float = Field(ge=0.0, le=1.0)
    is_flooded: bool = False
    roof_typology: Optional[str] = "rcc_slab"  # rcc_slab, tin_corrugated, terracotta_tile, thatch


class RoadSegment(BaseModel):
    segment_id: str
    status: RoadPassability
    blockage_cause: Optional[str] = None  # floodwater, debris, landslide_rubble
    bbox: BoundingBox


class GeoPoint(BaseModel):
    latitude: float
    longitude: float
    altitude_m: Optional[float] = 0.0


class TelemetryFrame(BaseModel):
    frame_id: str
    timestamp: float
    gps: GeoPoint
    altitude_agl_m: float
    heading_deg: float
    sensor: SensorModality = SensorModality.OPTICAL_RGB
    resolution_wh: Tuple[int, int] = (1920, 1080)
    image_path: Optional[str] = None


class UnifiedDisasterAnnotation(BaseModel):
    """Universal annotation container unifying all input formats."""
    sample_id: str
    dataset_source: str  # aider, xbd, floodnet, rescuenet, india_curated, synthetic
    disaster_class: DisasterClass
    confidence: float = 1.0
    overall_damage_grade: Optional[DamageGrade] = None
    buildings: List[BuildingFootprint] = Field(default_factory=list)
    roads: List[RoadSegment] = Field(default_factory=list)
    water_extent_pct: Optional[float] = 0.0
    burn_extent_pct: Optional[float] = 0.0
    telemetry: Optional[TelemetryFrame] = None
    vqa_pairs: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

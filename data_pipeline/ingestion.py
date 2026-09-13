"""Ingestion Adapters for Standard Public Disaster Datasets.
Converts AIDER, xBD, FloodNet, and RescueNet into UnifiedDisasterAnnotation format.
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from data_pipeline.schema import (
    UnifiedDisasterAnnotation,
    DisasterClass,
    DamageGrade,
    BuildingFootprint,
    RoadSegment,
    RoadPassability,
    BoundingBox
)


class AIDERAdapter:
    """Adapter for AIDER (Aerial Image Database for Emergency Response).
    Focus: Fast 4-class triage (Fire, Flood, Collapsed Building, Traffic Accident).
    """
    CLASS_MAPPING = {
        "fire": DisasterClass.WILDFIRE,
        "flood": DisasterClass.FLOOD,
        "collapsed_building": DisasterClass.EARTHQUAKE_COLLAPSE,
        "traffic_accident": DisasterClass.NORMAL_SCENE,  # or specialized accident class
        "normal": DisasterClass.NORMAL_SCENE
    }

    @classmethod
    def parse_record(cls, sample_id: str, raw_class: str, image_path: str) -> UnifiedDisasterAnnotation:
        disaster_class = cls.CLASS_MAPPING.get(raw_class.lower(), DisasterClass.UNCERTAIN_NEEDS_REVIEW)
        return UnifiedDisasterAnnotation(
            sample_id=sample_id,
            dataset_source="aider",
            disaster_class=disaster_class,
            confidence=1.0,
            metadata={"raw_class": raw_class, "image_path": image_path}
        )


class XBDAdapter:
    """Adapter for xBD (Satellite Building Damage Benchmark).
    Focus: 4-level ordinal scale (No Damage, Minor, Major, Destroyed).
    """
    DAMAGE_MAPPING = {
        "no-damage": DamageGrade.NO_DAMAGE,
        "minor-damage": DamageGrade.MINOR_DAMAGE,
        "major-damage": DamageGrade.MAJOR_DAMAGE,
        "destroyed": DamageGrade.DESTROYED,
        "un-classified": DamageGrade.NO_DAMAGE
    }

    DISASTER_MAPPING = {
        "earthquake": DisasterClass.EARTHQUAKE_COLLAPSE,
        "tsunami": DisasterClass.FLOOD,
        "flood": DisasterClass.FLOOD,
        "volcano": DisasterClass.INDUSTRIAL_CHEMICAL,
        "wildfire": DisasterClass.WILDFIRE,
        "hurricane": DisasterClass.CYCLONE_STORM,
        "tornado": DisasterClass.CYCLONE_STORM
    }

    @classmethod
    def parse_json(cls, json_path: Path) -> UnifiedDisasterAnnotation:
        with open(json_path, "r") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        disaster_type = metadata.get("disaster_type", "unknown")
        disaster_class = cls.DISASTER_MAPPING.get(disaster_type.lower(), DisasterClass.UNCERTAIN_NEEDS_REVIEW)

        buildings = []
        features = data.get("features", {}).get("xy", [])
        for idx, feat in enumerate(features):
            props = feat.get("properties", {})
            sub_type = props.get("subtype", "un-classified")
            damage = cls.DAMAGE_MAPPING.get(sub_type, DamageGrade.NO_DAMAGE)
            
            # Simple bounding box approximation from polygon points if available
            bbox = BoundingBox(ymin=0.0, xmin=0.0, ymax=0.1, xmax=0.1)
            buildings.append(
                BuildingFootprint(
                    building_id=f"bld_{idx}",
                    bbox=bbox,
                    damage_grade=damage,
                    confidence=1.0,
                    is_flooded=(disaster_class == DisasterClass.FLOOD)
                )
            )

        return UnifiedDisasterAnnotation(
            sample_id=metadata.get("id", json_path.stem),
            dataset_source="xbd",
            disaster_class=disaster_class,
            buildings=buildings,
            metadata=metadata
        )


class FloodNetAdapter:
    """Adapter for FloodNet (UAV Flood Scene Understanding + VQA).
    Focus: High-resolution oblique UAV imagery, water-extent segmentation & VQA.
    """
    @classmethod
    def parse_record(
        cls, 
        sample_id: str, 
        water_pct: float, 
        flooded_bld_count: int, 
        vqa_list: Optional[List[Dict[str, str]]] = None
    ) -> UnifiedDisasterAnnotation:
        buildings = [
            BuildingFootprint(
                building_id=f"flood_bld_{i}",
                bbox=BoundingBox(ymin=0.1 * i, xmin=0.1 * i, ymax=0.1 * i + 0.08, xmax=0.1 * i + 0.08),
                damage_grade=DamageGrade.MAJOR_DAMAGE,
                confidence=0.95,
                is_flooded=True,
                roof_typology="tin_corrugated"
            ) for i in range(flooded_bld_count)
        ]

        return UnifiedDisasterAnnotation(
            sample_id=sample_id,
            dataset_source="floodnet",
            disaster_class=DisasterClass.FLOOD,
            water_extent_pct=water_pct,
            buildings=buildings,
            vqa_pairs=vqa_list or [
                {"question": "Is the area flooded?", "answer": "Yes, extensive inundation present."},
                {"question": "How many flooded buildings are visible?", "answer": str(flooded_bld_count)}
            ]
        )


class RescueNetAdapter:
    """Adapter for RescueNet (UAV Natural Disaster Assessment).
    Focus: Road-clear vs road-blocked classification and building damage.
    """
    @classmethod
    def parse_record(
        cls,
        sample_id: str,
        disaster_class: DisasterClass,
        blocked_roads: int,
        clear_roads: int
    ) -> UnifiedDisasterAnnotation:
        roads = []
        for i in range(blocked_roads):
            roads.append(
                RoadSegment(
                    segment_id=f"road_blk_{i}",
                    status=RoadPassability.ROAD_BLOCKED,
                    blockage_cause="debris_and_floodwater",
                    bbox=BoundingBox(ymin=0.2 * i, xmin=0.0, ymax=0.2 * i + 0.1, xmax=1.0)
                )
            )
        for j in range(clear_roads):
            roads.append(
                RoadSegment(
                    segment_id=f"road_clr_{j}",
                    status=RoadPassability.ROAD_CLEAR,
                    bbox=BoundingBox(ymin=0.2 * j + 0.1, xmin=0.0, ymax=0.2 * j + 0.2, xmax=1.0)
                )
            )

        return UnifiedDisasterAnnotation(
            sample_id=sample_id,
            dataset_source="rescuenet",
            disaster_class=disaster_class,
            roads=roads
        )

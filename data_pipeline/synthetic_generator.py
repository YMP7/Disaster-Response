"""Synthetic Aerial Disaster Data Generator.
Generates procedural synthetic aerial imagery (nadir and oblique) with
realistic ground-truth annotations (building footprints, floodwaters,
structural debris, road blockages) to support zero-hardware simulation.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Any
from data_pipeline.schema import (
    UnifiedDisasterAnnotation,
    DisasterClass,
    DamageGrade,
    BuildingFootprint,
    RoadSegment,
    RoadPassability,
    BoundingBox,
    TelemetryFrame,
    GeoPoint
)


class SyntheticAerialGenerator:
    """Procedural generator creating photorealistic synthetic disaster scenes."""

    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.height, self.width = image_size

    def generate_scene(
        self,
        disaster_type: DisasterClass = DisasterClass.FLOOD,
        num_buildings: int = 6,
        seed: int = 42
    ) -> Tuple[np.ndarray, UnifiedDisasterAnnotation]:
        np.random.seed(seed)
        canvas = np.full((self.height, self.width, 3), 90, dtype=np.uint8)  # Base terrain

        # 1. Base terrain texture (vegetation / soil)
        noise = np.random.randint(-15, 15, (self.height, self.width, 3))
        canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)
        # Tint towards greenish-brown Indian rural/semi-urban soil
        canvas[..., 1] = np.clip(canvas[..., 1] * 1.1, 0, 255).astype(np.uint8)

        # 2. Roads
        roads = []
        road_y = int(self.height * 0.5)
        cv2.line(canvas, (0, road_y), (self.width, road_y), (50, 50, 55), 30)
        cv2.line(canvas, (0, road_y), (self.width, road_y), (220, 220, 220), 2)  # Center line

        road_status = RoadPassability.ROAD_CLEAR
        blockage_cause = None

        water_pct = 0.0
        buildings = []

        if disaster_type == DisasterClass.FLOOD:
            # Generate muddy floodwater polygon across lower half
            water_contour = np.array([
                [0, int(self.height * 0.4)],
                [int(self.width * 0.3), int(self.height * 0.45)],
                [int(self.width * 0.7), int(self.height * 0.38)],
                [self.width, int(self.height * 0.55)],
                [self.width, self.height],
                [0, self.height]
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [water_contour], (110, 130, 90))  # Silt-heavy brownish-green floodwater
            water_pct = 55.0
            road_status = RoadPassability.ROAD_BLOCKED
            blockage_cause = "monsoon_floodwater_inundation"

        elif disaster_type == DisasterClass.EARTHQUAKE_COLLAPSE:
            # Add ground fissure
            pts = np.array([[50, 100], [180, 220], [320, 310], [450, 480]], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (20, 20, 20), 4)
            road_status = RoadPassability.ROAD_BLOCKED
            blockage_cause = "structural_rubble_fissure"

        roads.append(
            RoadSegment(
                segment_id="road_nh_corridor",
                status=road_status,
                blockage_cause=blockage_cause,
                bbox=BoundingBox(
                    ymin=(road_y - 15) / self.height,
                    xmin=0.0,
                    ymax=(road_y + 15) / self.height,
                    xmax=1.0
                )
            )
        )

        # 3. Procedural Buildings (relative positions scaled to width/height)
        bld_w = int(self.width * 0.10)
        bld_h = int(self.height * 0.10)
        rel_positions = [
            (0.12, 0.12), (0.35, 0.15), (0.60, 0.12), (0.80, 0.18),
            (0.15, 0.65), (0.45, 0.70), (0.75, 0.65)
        ]

        for i in range(min(num_buildings, len(rel_positions))):
            rx, ry = rel_positions[i]
            bx = int(rx * self.width)
            by = int(ry * self.height)
            damage = DamageGrade.NO_DAMAGE
            is_flooded = False

            if disaster_type == DisasterClass.FLOOD and by > self.height * 0.38:
                damage = DamageGrade.MAJOR_DAMAGE
                is_flooded = True
                cv2.rectangle(canvas, (bx, by), (bx + bld_w, by + bld_h), (80, 95, 115), -1)
                cv2.line(canvas, (bx, by + (bld_h // 2)), (bx + bld_w, by + (bld_h // 2)), (140, 150, 110), 2)
            elif disaster_type == DisasterClass.EARTHQUAKE_COLLAPSE:
                if i % 2 == 0:
                    damage = DamageGrade.DESTROYED
                    for _ in range(8):
                        r_dx = bx + np.random.randint(-5, bld_w + 5)
                        r_dy = by + np.random.randint(-5, bld_h + 5)
                        cv2.rectangle(canvas, (r_dx, r_dy), (r_dx + 8, r_dy + 8), (140, 130, 125), -1)
                else:
                    damage = DamageGrade.MINOR_DAMAGE
                    cv2.rectangle(canvas, (bx, by), (bx + bld_w, by + bld_h), (160, 150, 140), -1)
            else:
                cv2.rectangle(canvas, (bx, by), (bx + bld_w, by + bld_h), (180, 175, 170), -1)
                cv2.circle(canvas, (bx + (bld_w // 4), by + (bld_h // 4)), 4, (40, 40, 40), -1)

            buildings.append(
                BuildingFootprint(
                    building_id=f"syn_bld_{i}",
                    bbox=BoundingBox(
                        ymin=min(1.0, max(0.0, by / self.height)),
                        xmin=min(1.0, max(0.0, bx / self.width)),
                        ymax=min(1.0, max(0.0, (by + bld_h) / self.height)),
                        xmax=min(1.0, max(0.0, (bx + bld_w) / self.width))
                    ),
                    damage_grade=damage,
                    confidence=0.98,
                    is_flooded=is_flooded,
                    roof_typology="rcc_slab" if i % 2 == 0 else "tin_corrugated"
                )
            )

        telemetry = TelemetryFrame(
            frame_id=f"sim_frame_{seed}",
            timestamp=1726000000.0 + seed,
            gps=GeoPoint(latitude=20.4625, longitude=85.8828, altitude_m=85.0), # Cuttack, Odisha
            altitude_agl_m=85.0,
            heading_deg=180.0
        )

        annotation = UnifiedDisasterAnnotation(
            sample_id=f"synthetic_{disaster_type.value}_{seed}",
            dataset_source="synthetic",
            disaster_class=disaster_type,
            confidence=0.99,
            buildings=buildings,
            roads=roads,
            water_extent_pct=water_pct,
            telemetry=telemetry
        )

        return canvas, annotation

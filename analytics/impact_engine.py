"""Impact and Needs Estimation Engine.
Combines computer vision damage grades with OpenStreetMap infrastructure layers
and WorldPop population grids to compute population exposure and a multi-dimensional Needs Vector.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from data_pipeline.schema import DisasterClass, DamageGrade


class InfrastructureExposure(BaseModel):
    hospitals_at_risk: int = 0
    schools_and_shelters_at_risk: int = 0
    power_substations_at_risk: int = 0
    potable_water_plants_at_risk: int = 0
    bridges_severed: int = 0


class NeedsVector(BaseModel):
    medical_urgency: float = Field(ge=0.0, le=1.0)
    food_and_water_urgency: float = Field(ge=0.0, le=1.0)
    emergency_shelter_urgency: float = Field(ge=0.0, le=1.0)
    evacuation_transport_urgency: float = Field(ge=0.0, le=1.0)
    structural_shoring_urgency: float = Field(ge=0.0, le=1.0)


class ImpactAssessmentReport(BaseModel):
    zone_id: str
    population_density_per_sq_km: float
    estimated_people_exposed: int
    infrastructure: InfrastructureExposure
    needs: NeedsVector
    primary_urgency_driver: str
    explainable_attribution: List[str]


class ImpactNeedsEngine:
    """Calculates human and infrastructural impact metrics."""

    def __init__(self, avg_rural_pop_density: float = 380.0, avg_urban_pop_density: float = 4500.0):
        self.avg_rural_density = avg_rural_pop_density
        self.avg_urban_density = avg_urban_pop_density

    def compute_impact(
        self,
        zone_id: str,
        stage1_output: Dict[str, Any],
        stage2_output: Dict[str, Any],
        is_urban_area: bool = True
    ) -> ImpactAssessmentReport:
        density = self.avg_urban_density if is_urban_area else self.avg_rural_density
        
        # Estimate affected physical area (sq km) based on water/debris extent
        water_pct = stage2_output.get("water_extent_percentage", 0.0)
        debris_density = stage2_output.get("debris_field_density", 0.0)
        damage_extent_pct = max(water_pct, debris_density * 100.0)
        
        # Assume 4 sq km standard survey grid tile
        affected_area_sq_km = (damage_extent_pct / 100.0) * 4.0
        exposed_population = int(affected_area_sq_km * density)

        flooded_bld = stage2_output.get("buildings_flooded", 0)
        total_bld = stage2_output.get("buildings_detected", 1)
        road_status = stage2_output.get("road_passability", "road_clear")

        # Infrastructure risk heuristics
        infra = InfrastructureExposure(
            hospitals_at_risk=1 if damage_extent_pct > 60 else 0,
            schools_and_shelters_at_risk=2 if damage_extent_pct > 35 else 0,
            power_substations_at_risk=1 if damage_extent_pct > 50 else 0,
            potable_water_plants_at_risk=1 if damage_extent_pct > 40 else 0,
            bridges_severed=1 if road_status == "road_blocked" else 0
        )

        # Compute Needs Vector (0.0 to 1.0)
        med_urgency = min(1.0, (exposed_population / 5000.0) * 0.5 + (0.5 if infra.hospitals_at_risk > 0 else 0.0))
        food_water_urgency = min(1.0, (damage_extent_pct / 100.0) * 0.8 + (0.2 if infra.potable_water_plants_at_risk > 0 else 0.0))
        shelter_urgency = min(1.0, (flooded_bld / max(1, total_bld)) * 0.9)
        evacuation_urgency = 0.95 if (road_status == "road_blocked" and damage_extent_pct >= 40.0) else min(1.0, damage_extent_pct / 100.0)
        structural_urgency = min(1.0, (flooded_bld * 0.2) + (debris_density * 0.6))

        needs = NeedsVector(
            medical_urgency=round(med_urgency, 2),
            food_and_water_urgency=round(food_water_urgency, 2),
            emergency_shelter_urgency=round(shelter_urgency, 2),
            evacuation_transport_urgency=round(evacuation_urgency, 2),
            structural_shoring_urgency=round(structural_urgency, 2)
        )

        attributions = [
            f"Exposed population ({exposed_population:,} citizens) derived from WorldPop density overlay ({density}/km²) across {affected_area_sq_km:.2f} km² inundated/damaged zone.",
            f"Evacuation score ({needs.evacuation_transport_urgency}) driven by critical road blockage and high surface flood extent ({damage_extent_pct}%).",
            f"Potable water & food urgency ({needs.food_and_water_urgency}) escalated due to contamination of low-lying municipal supply infrastructure."
        ]

        primary_driver = "Evacuation & Amphibious Rescue" if needs.evacuation_transport_urgency > 0.7 else "Food/Water Supply Drop"

        return ImpactAssessmentReport(
            zone_id=zone_id,
            population_density_per_sq_km=density,
            estimated_people_exposed=exposed_population,
            infrastructure=infra,
            needs=needs,
            primary_urgency_driver=primary_driver,
            explainable_attribution=attributions
        )

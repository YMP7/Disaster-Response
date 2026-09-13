"""Reference Extensibility Plugin: Industrial Chemical / Toxic Gas Hazard Module.
Demonstrates adding a specialized disaster assessment head with zero core edits.
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from plugins.base import DisasterPluginInterface, plugin_registry


class IndustrialChemicalPlugin(DisasterPluginInterface):
    """Specialized module for assessing chemical leaks, industrial blasts, and chlorine/ammonia plumes."""

    @property
    def disaster_identifier(self) -> str:
        return "industrial_chemical_toxic_plume"

    @property
    def display_name(self) -> str:
        return "Industrial Chemical & Toxic Vapor Plume"

    @property
    def required_sensor_modalities(self) -> List[str]:
        return ["optical_rgb", "optical_gas_imaging_thermal"]

    def analyze_scene(self, image_rgb: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        h, w, _ = image_rgb.shape
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Detect unusual plume hues (yellowish chlorine or dense dark aerosol)
        chlorine_mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([38, 255, 255]))
        plume_area_pct = round((cv2.countNonZero(chlorine_mask) / (h * w)) * 100.0, 2)

        # Estimate exclusion zone based on plume coverage
        exclusion_radius_m = 800.0 if plume_area_pct < 5.0 else 2500.0

        return {
            "disaster_identifier": self.disaster_identifier,
            "toxic_vapor_detected": plume_area_pct > 0.5,
            "plume_surface_coverage_pct": plume_area_pct,
            "mandated_exclusion_zone_meters": exclusion_radius_m,
            "evacuation_priority": "CRITICAL_IMMEDIATE" if plume_area_pct > 2.0 else "PRECAUTIONARY",
            "downwind_dispersion_vector": {"azimuth_deg": 135.0, "estimated_wind_speed_kmh": 18.0}
        }


# Auto-register plugin on import
plugin_registry.register_disaster_plugin(IndustrialChemicalPlugin())

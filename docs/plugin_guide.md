# Developer & Extensibility Guide: India Disaster Response AI Platform

This guide outlines how to extend the platform with **new disaster types**, **new sensor modalities**, and **new drone autopilots** without modifying core codebase files or breaking existing deployments.

---

## 1. Adding a New Disaster Type

To add a new disaster classification and assessment head (for example, a **Himalayan Glacial Lake Outburst Flood (GLOF)** or an **Industrial Gas Leak**):

### Step 1: Define the Plugin Class
Inherit from `plugins.base.DisasterPluginInterface`:

```python
# plugins/examples/glof_plugin.py
import numpy as np
from plugins.base import DisasterPluginInterface, plugin_registry

class HimalayanGLOFPlugin(DisasterPluginInterface):
    @property
    def disaster_identifier(self) -> str:
        return "himalayan_glof"

    @property
    def display_name(self) -> str:
        return "Glacial Lake Outburst Flood (GLOF)"

    @property
    def required_sensor_modalities(self) -> list:
        return ["optical_rgb", "dem_elevation"]

    def analyze_scene(self, image_rgb: np.ndarray, metadata: dict) -> dict:
        # Specialized moraine dam breach & sediment rush analysis
        return {
            "disaster_identifier": self.disaster_identifier,
            "moraine_breach_detected": True,
            "downstream_flood_wave_velocity_mps": 14.5,
            "valley_evacuation_lead_time_min": 25.0
        }

# Auto-register plugin
plugin_registry.register_disaster_plugin(HimalayanGLOFPlugin())
```

### Step 2: Register in `config/disaster_registry.json`
Add the configuration block so the Stage-1 triage router knows how to dispatch to it:

```json
{
  "himalayan_glof": {
    "display_name": "Glacial Lake Outburst Flood",
    "visual_cues": "Moraine wall breach, high-velocity gray sediment flow in mountain gorge",
    "stage2_module": "plugins.examples.glof_plugin.HimalayanGLOFPlugin",
    "priority_level": 1,
    "sensor_modalities": ["optical_rgb", "dem_elevation"],
    "response_playbook": "downstream_valley_siren_and_hydro_shutdown"
  }
}
```

---

## 2. Adding a New Sensor Modality (e.g. Thermal FLIR or LiDAR)

To ingest and calibrate non-RGB sensors:

### Step 1: Inherit from `SensorPluginInterface`
```python
# plugins/sensors/thermal_flir.py
import numpy as np
from plugins.base import SensorPluginInterface, plugin_registry

class ThermalFLIRPlugin(SensorPluginInterface):
    @property
    def sensor_name(self) -> str:
        return "thermal_flir_radiometric"

    def preprocess_raw_data(self, raw_bytes: bytes) -> np.ndarray:
        # Convert 16-bit radiometric Kelvin arrays to normalized Celsius temperature maps
        raw_16bit = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((512, 640))
        celsius = (raw_16bit * 0.04) - 273.15
        return celsius

plugin_registry.register_sensor_plugin(ThermalFLIRPlugin())
```

---

## 3. Adding a New Drone Autopilot Backend (e.g. ROS2 / Micro-XRCE)

The platform comes default with MAVLink/MAVSDK contracts. If your fleet runs native ROS2 (e.g. PX4 ROS2 bridge or custom companion computers):

```python
# plugins/backends/ros2_backend.py
from plugins.base import DroneBackendInterface, plugin_registry

class ROS2AutopilotBackend(DroneBackendInterface):
    @property
    def backend_name(self) -> str:
        return "ROS2_PX4_MicroXRCE"

    def dispatch_mission(self, mission_contract_dict: dict) -> bool:
        # Publish trajectory setpoint messages over ROS2 topics
        print(f"[ROS2] Publishing mission {mission_contract_dict['mission_id']} to /fmu/in/trajectory_setpoint")
        return True

plugin_registry.register_drone_backend(ROS2AutopilotBackend())
```

---

## 4. Model Versioning & Dynamic Rollback

When retraining models (e.g., adding Kerala flood annotations or fine-tuning MobileNetV3):

1. **Train and register new weights**:
   ```python
   from models.registry import ModelRegistry
   from pathlib import Path

   registry = ModelRegistry()
   registry.register_model(
       model_id="stage1_mobilenetv3_india_v2",
       version="2.0.0",
       stage="stage1_triage",
       architecture="MobileNetV3-Small",
       dataset_provenance=["AIDER", "IndiaCuratedSet", "FloodNet"],
       weights_path=Path("models/weights/stage1_v2.pt"),
       metrics={"val_accuracy": 0.942, "ece": 0.041},
       activate_immediately=True
   )
   ```

2. **Rollback if field regression occurs**:
   ```python
   # Instant rollback without touching code or redeploying containers
   registry.rollback(stage="stage1_triage", target_version="1.0.0")
   ```

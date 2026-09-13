"""Extensibility Plugin Interfaces and Dynamic Registry.
Enables developers to add:
1. New Disaster Types (e.g. Glacial Lake Outburst Floods, Dam Bursts)
2. New Sensor Modalities (e.g. Thermal FLIR, Hyperspectral, LiDAR)
3. New Autopilot Backends (e.g. ROS2, PX4 uORB, DJI Mobile SDK)
without modifying or rearchitecting core platform code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
import numpy as np
from pydantic import BaseModel


class DisasterPluginInterface(ABC):
    """Interface for registering a new disaster classification and assessment head."""

    @property
    @abstractmethod
    def disaster_identifier(self) -> str:
        """Unique identifier, e.g. 'himalayan_glof', 'chemical_plume'."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        pass

    @property
    @abstractmethod
    def required_sensor_modalities(self) -> List[str]:
        pass

    @abstractmethod
    def analyze_scene(self, image_rgb: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the specialized Stage-2 evaluation."""
        pass


class SensorPluginInterface(ABC):
    """Interface for ingesting non-standard sensor channels."""

    @property
    @abstractmethod
    def sensor_name(self) -> str:
        pass

    @abstractmethod
    def preprocess_raw_data(self, raw_bytes: bytes) -> np.ndarray:
        pass


class DroneBackendInterface(ABC):
    """Interface for connecting to alternative drone autopilots or middleware."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """e.g. 'MAVLink_v2', 'ROS2_Humble', 'Auterion_API'."""
        pass

    @abstractmethod
    def dispatch_mission(self, mission_contract_dict: Dict[str, Any]) -> bool:
        pass


class GlobalPluginRegistry:
    """Dynamic registry holding loaded plugins."""

    def __init__(self):
        self.disaster_plugins: Dict[str, DisasterPluginInterface] = {}
        self.sensor_plugins: Dict[str, SensorPluginInterface] = {}
        self.drone_backends: Dict[str, DroneBackendInterface] = {}

    def register_disaster_plugin(self, plugin: DisasterPluginInterface):
        self.disaster_plugins[plugin.disaster_identifier] = plugin
        print(f"[PluginRegistry] Registered disaster module: {plugin.disaster_identifier} ({plugin.display_name})")

    def register_sensor_plugin(self, plugin: SensorPluginInterface):
        self.sensor_plugins[plugin.sensor_name] = plugin
        print(f"[PluginRegistry] Registered sensor module: {plugin.sensor_name}")

    def register_drone_backend(self, backend: DroneBackendInterface):
        self.drone_backends[backend.backend_name] = backend
        print(f"[PluginRegistry] Registered drone autopilot backend: {backend.backend_name}")

    def get_disaster_head(self, disaster_id: str) -> Optional[DisasterPluginInterface]:
        return self.disaster_plugins.get(disaster_id)


# Global Plugin Registry Singleton
plugin_registry = GlobalPluginRegistry()

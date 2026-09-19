"""Core Configuration and Settings for India Disaster Response Platform.
Implements statutory norms from Ministry of Home Affairs (MHA) SDRF/NDRF guidelines
and airspace boundaries per DGCA Drone Rules 2021.
"""

from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from config.version import __version__

BASE_DIR = Path(__file__).resolve().parent.parent

class SDRFNorms(BaseModel):
    """Statutory State/National Disaster Response Fund compensation rates (INR).
    Based on official Ministry of Home Affairs (MHA) disaster relief schedules.
    """
    # Housing Assistance
    fully_destroyed_rcc: float = 120_000.0   # Pucca / RCC houses in plains
    fully_destroyed_hilly: float = 130_000.0 # Pucca in hilly / difficult terrain
    fully_destroyed_kutcha: float = 80_000.0 # Kutcha / mud / thatch
    severely_damaged_pucca: float = 40_000.0
    severely_damaged_kutcha: float = 10_000.0
    partially_damaged_pucca: float = 6_500.0
    partially_damaged_kutcha: float = 4_000.0

    # Agriculture & Livelihood Relief (per hectare / unit)
    crop_loss_rainfed_per_ha: float = 8_500.0
    crop_loss_irrigated_per_ha: float = 17_000.0
    perennial_crops_per_ha: float = 22_500.0
    milch_animal_loss_per_unit: float = 37_500.0

    # Immediate Relief
    clothing_and_utensils_per_family: float = 5_000.0
    gratuitous_relief_per_adult_day: float = 150.0

class DGCARules(BaseModel):
    """DGCA Drone Rules 2021 Operational Airspace Limits."""
    max_altitude_agl_meters: float = 120.0  # 400 ft Above Ground Level (AGL)
    green_zone_max_altitude_meters: float = 120.0
    yellow_zone_controlled: bool = True     # Requires Air Traffic Control clearance
    red_zone_prohibited: bool = True        # Strictly prohibited without MoCA permission
    npnt_enforced: bool = True              # No Permission No Takeoff compliance
    rpic_override_required: bool = True     # Remote Pilot in Command always maintains kill-switch

class Settings(BaseModel):
    project_name: str = "DisasterResponseAI-India"
    version: str = __version__
    base_dir: Path = BASE_DIR
    
    # Storage & Data Paths
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models" / "weights"
    audit_log_path: Path = BASE_DIR / "audit_trail.jsonl"
    
    # Model Thresholds
    confidence_threshold: float = 0.70      # Below this, route to UNCERTAIN_NEEDS_REVIEW
    entropy_threshold: float = 1.25         # High distribution entropy indicates ambiguity
    
    # Live Monitoring
    monitored_radius_km: float = 75.0
    sachet_polling_interval_sec: int = 30
    
    
    # Drone Communication & Autopilot Interface
    # Safe default: MOCK_HARNESS (zero network side-effects)
    # Options: "MOCK_HARNESS", "SITL_UDP", "PX4_PHYSICAL"
    drone_comm_mode: str = "MOCK_HARNESS"
    mavlink_connection_str: str = "udpin:0.0.0.0:14550"
    mavlink_target_system: int = 1
    mavlink_target_component: int = 1
    mavlink_heartbeat_timeout_sec: float = 5.0
    mavlink_mission_ack_timeout_sec: float = 5.0

    # Norms and Airspace
    sdrf_rates: SDRFNorms = Field(default_factory=SDRFNorms)
    dgca_rules: DGCARules = Field(default_factory=DGCARules)
    
    # Supported Languages (SACHET-aligned)
    supported_languages: List[str] = [
        "en",    # English
        "hi",    # Hindi
        "or",    # Odia (Cyclone/Flood prone)
        "ml",    # Malayalam (Kerala Floods)
        "bn",    # Bengali (Sundarbans / West Bengal)
        "ta",    # Tamil (Coastal Tamil Nadu)
        "te",    # Telugu (Andhra Coastal)
    ]

settings = Settings()

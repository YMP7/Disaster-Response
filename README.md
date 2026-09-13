# Disaster Response AI Platform (India)
### Autonomous Disaster Detection, Impact Analysis & Simulator-First Drone Response System

[![Tests: 17 Passed](https://img.shields.io/badge/Tests-17%20Passed%20(100%25)-success)](tests/test_end_to_end.py)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Regulatory: DGCA Drone Rules 2021](https://img.shields.io/badge/Compliance-DGCA%20Drone%20Rules%202021-orange)](docs/architecture.md)
[![Privacy: DPDP Act 2023](https://img.shields.io/badge/Privacy-India%20DPDP%20Act%202023-purple)](data_pipeline/privacy_filter.py)

---

## Overview

The **Disaster Response AI Platform** is an autonomous, safety-critical decision-support system designed specifically for the operational, infrastructural, and regulatory environment of India. 

The platform ingests multi-modal disaster imagery (satellite, UAV, CCTV, crowdsourced), executes multi-stage computer vision triage and damage grading, continuously monitors national warning feeds (**NDMA SACHET / Oasis CAP v1.2**), coordinates a simulator-backed **MAVLink** drone abstraction layer with on-scene re-analysis, computes explainable relief fund allocations grounded in **Ministry of Home Affairs (MHA) State Disaster Response Fund (SDRF)** schedules, and enforces mandatory **Human-in-the-Loop (HITL)** gates and **DPDP** privacy protections.

---

## Core Architectural Axioms

1. **Protocol-Agnostic, Simulator-First Autopilot**: Emits standardized MAVLink/MAVSDK mission contracts (`MISSION_ITEM_INT`). Operates identically in PX4 SITL / Gazebo simulation today and on physical companion computers (NVIDIA Jetson) tomorrow with zero business logic modification.
2. **Two-Stage Hierarchical Computer Vision**: Never runs expensive segmentation blindly. A sub-50ms Stage-1 edge classifier triages disaster type first before routing to specialized Stage-2 ordinal damage and extent heads.
3. **No False Certainty (Sensor Disagreement Mitigated)**: Grounded in research showing satellite and drone damage labels frequently diverge, predictions carry calibrated confidence and an explicit `UNCERTAIN_NEEDS_REVIEW` routing state for low-confidence or high-entropy samples.
4. **Separation of Statutory Law vs. Algorithmic Optimization**: Compensation currency values are **never learned by black-box regression models**. Relief calculations use deterministic, auditable lookup tables matching official MHA SDRF/NDRF schedules. Optimization is strictly applied to spatial sequencing and claim fraud/divergence detection.
5. **Mandatory Human-in-the-Loop (HITL) Gate**: No irreversible action—disbursing public funds, launching flight payloads, or redirecting emergency responders—can execute without affirmative officer sign-off.
6. **Cryptographic Auditability & Privacy**: Every event is recorded in an immutable, append-only SHA-256 hash-chained ledger (`audit_trail.jsonl`). Facial identities and vehicle license plates are automatically blurred before storage per India's Digital Personal Data Protection (DPDP) Act.

---

## System Architecture

```
                                  PUBLIC EARLY-WARNING FEEDS
                    ┌─────────────────────────────────────────────────────┐
                    │  NDMA SACHET (Oasis CAP v1.2 Feeds):                │
                    │  - IMD (Cyclone / Severe Weather)                   │
                    │  - CWC (Central Water Commission - River Flood)    │
                    │  - INCOIS (Tsunami / Storm Surge Alerts)            │
                    │  - NRSC / Bhuvan (Satellite Inundation Layers)      │
                    └──────────────────────────┬──────────────────────────┘
                                               │ Polling / Push
                                               ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ 1. LIVE MONITORING & GEOFENCING SERVICE             │
                    │    - Configurable Center-Point + Monitored Radius   │
                    │    - Great-Circle Haversine & Spatial Intersection  │
                    │    - Deduplication & Severity Escalation Engine     │
                    └──────────────────────────┬──────────────────────────┘
                                               │ TypedAlertEvent
                                               ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ 2. DISASTELLER MULTI-AGENT ORCHESTRATION LAYER      │
                    │    - Detection Agent (Stage-1 Edge Triage)          │
                    │    - Assessment Agent (Stage-2 Damage Heads)        │
                    │    - Resource & Fund Agent (SDRF Norms & Fraud)     │
                    │    - Multilingual Reporting Agent (SACHET 7-Lang)   │
                    └───────────┬───────────────────────────────┬─────────┘
                                │                               │
             Aerial Imagery     │                               │ Mission Intent
                                ▼                               ▼
┌───────────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
│ 3. COMPUTER VISION & REASONING PIPELINE       │ │ 4. DRONE MISSION ABSTRACTION LAYER       │
│                                               │ │                                          │
│ ┌───────────────────────────────────────────┐ │ │ ┌──────────────────────────────────────┐ │
│ │ Stage 1: Fast Edge Classifier (AIDER)     │ │ │ │ DGCA Drone Rules 2021 Safety Checker │ │
│ │ - MobileNetV3 / Sub-50ms Inference        │ │ │ │ - Green/Yellow/Red Airspace Polygons │ │
│ │ - Triage: Flood, Cyclone, Quake, Fire, etc│ │ │ │ - 120m (400 ft) AGL Ceiling Enforcer │ │
│ └─────────────────────┬─────────────────────┘ │ │ │ - NPNT (No Permission No Takeoff) Gate│ │
│                       │                       │ │ └──────────────────┬───────────────────┘ │
│                       ▼                       │ │                    │ Cleared Mission     │
│ ┌───────────────────────────────────────────┐ │ │                    ▼                     │
│ │ Stage 2: Specialized Severity & Extent    │ │ │ ┌──────────────────────────────────────┐ │
│ │ - xBD 4-Level Ordinal Damage Grading      │ │ │ │ Hardware-Agnostic Mission Contract   │ │
│ │ - FloodNet Water / Building Segmentation  │ │ │ │ (Waypoints, Altitude, Failsafes)     │ │
│ │ - RescueNet Road Passability (Clear/Block)│ │ │ └──────────────────┬───────────────────┘ │
│ │ - BRIGHT Multimodal Optical+SAR Fusion    │ │ │                    │                     │
│ └─────────────────────┬─────────────────────┘ │ │                    ▼                     │
│                       │                       │ │ ┌──────────────────────────────────────┐ │
│                       ▼                       │ │ │ PX4 SITL / MAVLink Flight Simulator  │ │
│ ┌───────────────────────────────────────────┐ │ │ │ - Telemetry & Battery Depletion Stream│ │
│ │ Stage 3: Multilingual VLM Reasoning       │ │ │ │ - Simulated Aerial Video Stream      │ │
│ │ - Advisory Sit-Reps: EN, HI, OR, ML, etc. │ │ │ └──────────────────┬───────────────────┘ │
│ │ - Grounded VQA (Cannot override CV data)  │ │ │                    │                     │
│ └─────────────────────┬─────────────────────┘ │ │                    ▼                     │
│                       │                       │ │ ┌──────────────────────────────────────┐ │
│                       ▼                       │ │ │ On-Scene Re-Analysis Loop            │ │
│ ┌───────────────────────────────────────────┐ │ │ │ - Multi-Altitude Visual Inspection   │ │
│ │ Model Registry & Rollback Engine          │ │ │ │ - Constrained Action Enum Decision   │ │
│ │ - SHA-256 Verification & A/B Pointers     │ │ │ └──────────────────┬───────────────────┘ │
│ │ - Calibration: ECE & Entropy Thresholding │ │ └────────────────────┼─────────────────────┘
│ └─────────────────────┬─────────────────────┘                        │
└───────────────────────┼──────────────────────────────────────────────┘
                        │ Quantitative Damage Metrics & Road Network Status
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. IMPACT & NEEDS ENGINE                                                               │
│    - Population Exposure: WorldPop Gridded Density Overlay                              │
│    - Critical Infrastructure: Hospitals, Substations, Bridges, Water Treatment         │
│    - Multi-Dimensional Needs Vector: [Medical, Food/Water, Shelter, Evacuation]        │
└───────────────────────────────────────┬────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. EXPLAINABLE SDRF / NDRF FUND ALLOCATION & ANOMALY DETECTION                         │
│    - Ministry of Home Affairs Statutory Compensation Rates (Pucca / Kutcha / Cropland) │
│    - Itemized Mathematical Ledger: Plain-language feature attributions per Rupee       │
│    - Claim Discrepancy & Fraud Detector: Flags divergence between claims and imagery   │
└───────────────────────────────────────┬────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 7. SAFETY-CRITICAL COMMAND CENTER & HUMAN-IN-THE-LOOP (HITL) GATE                      │
│    - Mandatory Human Sign-Off on Drone Takeoff, Relief Tranches & Crew Redirection     │
│    - Automated DPDP Redaction: Real-time Gaussian masking of faces & vehicle plates    │
│    - Immutable Cryptographic Audit Log: Append-only SHA-256 chained transaction trail  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```text
.
├── config/
│   ├── settings.py                 # Core configurations, MHA SDRF rates & DGCA parameters
│   ├── disaster_registry.json     # Dynamic registry for disaster types & stage-2 modules
│   └── model_registry.json        # Active model versions, SHA-256 checksums & rollback
├── data_pipeline/
│   ├── schema.py                   # Pydantic schemas (AIDER, xBD, FloodNet, RescueNet)
│   ├── ingestion.py                # Public benchmark ingestion adapters
│   ├── india_augmentation.py       # Monsoon turbidity, rain streaks & Indian roof typologies
│   ├── privacy_filter.py           # DPDP Act automated face and vehicle plate redaction
│   ├── synthetic_generator.py      # Zero-hardware procedural aerial scene generator
│   └── india_curated_set.py        # Curated held-out benchmark set (Kerala, Odisha, Chamoli)
├── models/
│   ├── registry.py                 # Model registry with SHA-256 verification & rollback
│   ├── multimodal_fusion.py        # BRIGHT-inspired Optical + SAR cross-attention fusion
│   ├── stage1_classifier.py        # MobileNetV3 lightweight edge triage classifier
│   ├── stage2_severity.py          # xBD ordinal damage & FloodNet extent segmentation heads
│   ├── calibration.py              # Temperature scaling & explicit uncertainty gating (<70% conf)
│   ├── stage3_reasoning_vlm.py     # Multilingual advisory VLM reporting (EN, HI, OR, ML, etc.)
│   └── train.py                    # Training & ECE calibration evaluation harness
├── live_monitoring/
│   ├── cap_parser.py               # Oasis CAP v1.2 XML/JSON parser for NDMA SACHET
│   ├── geofence.py                 # Great-circle Haversine & radius geofence evaluator
│   └── alert_service.py            # Event loop dispatching TypedAlertEvent notifications
├── drone_abstraction/
│   ├── protocol.py                 # Protocol-agnostic MAVLink/MAVSDK mission contract
│   ├── dgca_compliance.py          # DGCA Drone Rules 2021 checker (120m ceiling & Red Zones)
│   ├── simulator.py                # PX4 SITL flight simulator streaming telemetry & camera frames
│   └── on_scene_agent.py           # On-scene aerial agent with constrained action enum
├── analytics/
│   ├── impact_engine.py            # WorldPop density overlay & multi-dimensional Needs Vector
│   ├── fund_allocation.py          # Statutory MHA SDRF/NDRF relief fund engine with explainability
│   └── anomaly_detector.py         # Municipal claim fraud & imagery discrepancy detector
├── orchestration/
│   ├── audit_log.py                # Immutable append-only SHA-256 cryptographic audit trail
│   ├── hitl_gate.py                # Mandatory Human-in-the-Loop command approval gate
│   ├── agents.py                   # DisasTeller multi-agent backend (Detection, Assessment, Funds, Reporting)
│   └── api.py                      # FastAPI REST & event endpoints
├── plugins/
│   ├── base.py                     # Extensibility plugin interfaces for disasters, sensors & autopilots
│   └── examples/                   # Reference Industrial Chemical / Toxic Vapor hazard plugin
├── docs/
│   ├── architecture.md             # Complete system architecture specification
│   ├── plugin_guide.md             # Developer guide to adding new disasters, sensors & autopilots
│   └── validation_plan.md          # 3-tier zero-hardware validation plan
├── tests/
│   └── test_end_to_end.py          # 17 automated unit and integration tests (100% passing)
├── main.py                         # End-to-end incident lifecycle simulation script
└── README.md                       # Platform documentation
```

---

## Disaster Taxonomy & Routing

The Stage-1 classifier routes raw incoming imagery into specialized Stage-2 analytical modules:

| Disaster Class | Visual Cues | Stage-2 Module Focus | Regulatory Priority |
| :--- | :--- | :--- | :---: |
| **Flood** | Silt turbidity, water extent, submerged roads | Flood extent segmentation, flooded vs. dry structures (FloodNet) | 1 |
| **Cyclone / Storm** | Roof loss, debris scatter, fallen palms | Debris field density, unroofed structures | 1 |
| **Earthquake / Collapse** | Structural tilt, pancake collapse, rubble | 4-level ordinal damage grading (xBD scale) | 1 |
| **Landslide** | Slope scarring, debris flow, road cuts | Scar area %, transportation artery blockage | 2 |
| **Wildfire** | Smoke plume, burn scars, active flames | Flame perimeter, smoke dispersion rate | 2 |
| **Industrial / Chemical** | Chemical smoke plume, blast damage | Toxic vapor dispersion, exclusion zone radius | 1 |
| **Drought / Heatwave** | NDVI vegetation decline, reservoir shrinkage | Trend-based time-series anomaly | 3 |
| **Normal Scene** | Intact structures, standard traffic | Normal monitoring (no action) | 4 |
| **Uncertain / Review** | Low confidence ($<0.70$), high entropy ($>1.25$) | Escalate directly to Command Center Human Gate | 1 |

---

## Statutory Relief Funding (MHA SDRF/NDRF Norms)

The platform encodes official **Ministry of Home Affairs (MHA)** State Disaster Response Fund schedules as deterministic lookup tables:

- **Fully Destroyed Pucca (RCC) House (Plains)**: ₹1,20,000 / unit
- **Fully Destroyed Pucca (Hilly / Difficult Terrain)**: ₹1,30,000 / unit
- **Fully Destroyed Kutcha House**: ₹80,000 / unit
- **Severely Damaged Pucca House**: ₹40,000 / unit
- **Severely Damaged Kutcha House**: ₹10,000 / unit
- **Agricultural Crop Loss (Rainfed, >33% Damage)**: ₹8,500 / hectare
- **Agricultural Crop Loss (Irrigated, >33% Damage)**: ₹17,000 / hectare
- **Loss of Clothing & Utensils**: ₹5,000 / affected family

### Discrepancy & Fraud Detector
Compares aerial computer vision damage indices against self-reported municipal/public loss claims. If claims diverge by $>40\%$, the platform flags the claim and issues audit recommendations:
- **Discrepancy $>2.5\times$**: `CRITICAL RISK` $\rightarrow$ Hold disbursement; mandate physical re-survey by State Vigilance Commission.
- **Discrepancy $1.4\times - 2.5\times$**: `HIGH RISK` $\rightarrow$ Flag for reconciliation against GIS property tax records.
- **Discrepancy $<0.6\times$**: `MEDIUM RISK` $\rightarrow$ Potential under-reporting; dispatch ground enumerators to protect vulnerable households.

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.0+ & Torchvision
- OpenCV, NumPy, Pydantic v2, FastAPI, Rich, Pytest

```bash
# Clone the repository
git clone https://github.com/YMP7/Disaster-Response.git
cd Disaster-Response

# Install dependencies
pip install torch torchvision opencv-python pydantic fastapi uvicorn rich pytest
```

### 2. Run the Full Test Suite
Runs all 17 unit and integration tests covering data pipelines, CV models, SAR fusion, calibration, aviation rules, simulator, SDRF math, and cryptographic audit logs:

```bash
python -m pytest -v tests/test_end_to_end.py
```

### 3. Run the End-to-End Incident Lifecycle Simulation
Executes a complete simulated incident (SACHET alert $\rightarrow$ Geofence match $\rightarrow$ DGCA safety audit $\rightarrow$ PX4 SITL flight $\rightarrow$ On-scene CV $\rightarrow$ DPDP privacy blurring $\rightarrow$ SDRF relief funding $\rightarrow$ Fraud check $\rightarrow$ Multilingual sit-reps $\rightarrow$ HITL sign-off):

```bash
python main.py
```

### 4. Start the REST API Service
Launches the FastAPI backend exposing monitoring, mission planning, and HITL review endpoints:

```bash
uvicorn orchestration.api:app --reload --port 8000
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

---

## Extensibility & Plugins

The platform is built with a plugin registry pattern allowing zero-code-change additions:

1. **New Disaster Type**: Inherit from `plugins.base.DisasterPluginInterface` and register in `config/disaster_registry.json`.
2. **New Sensor Modality**: Inherit from `plugins.base.SensorPluginInterface` (e.g. Thermal FLIR, Hyperspectral Gas Imagers).
3. **New Autopilot Backend**: Inherit from `plugins.base.DroneBackendInterface` (e.g. ROS2 Humble, Micro-XRCE-DDS, Auterion).

See [docs/plugin_guide.md](docs/plugin_guide.md) for detailed code examples and integration recipes.

---

## Literature & Regulatory Grounding

| # | Paper / Regulatory Standard | Architectural Component Justified |
| :---: | :--- | :--- |
| 1 | **Kyrkou & Theocharides (CVPRW 2019)** [arXiv:1906.08716](https://arxiv.org/pdf/1906.08716) | AIDER aerial triage taxonomy & two-stage routing. |
| 2 | **EmergencyNet (IEEE JSTARS 2021)** [arXiv:2104.14006](https://arxiv.org/pdf/2104.14006) | Lightweight CNN backbones (MobileNetV3) for embedded edge inference. |
| 3 | **Younis et al. (2018)** [arXiv:1807.11805](https://arxiv.org/pdf/1807.11805) | Minimalist multi-disaster baseline evaluation. |
| 4 | **Alsaaran & Soudani (Sensors 2025)** [doi:10.3390/s25175406](https://doi.org/10.3390/s25175406) | Real-time edge structural damage level prediction. |
| 5 | **Evaluating Fine-Tuned DL (Springer 2024)** [doi:10.1007/s43503-024-00034-6](https://link.springer.com/article/10.1007/s43503-024-00034-6) | Selection of segmentation vs. detection heads for collapsed structures. |
| 6 | **xBD Dataset (Gupta et al., 2019)** [arXiv:1911.09296](https://arxiv.org/pdf/1911.09296) | Standardized 4-level ordinal damage scale (`No`, `Minor`, `Major`, `Destroyed`). |
| 7 | **FloodNet (Rahnemoonfar et al., 2021)** [arXiv:2012.02951](https://arxiv.org/pdf/2012.02951) | High-resolution UAV oblique flood extent segmentation and VQA. |
| 8 | **RescueNet (Rahnemoonfar et al., 2023)** [Nature Sci Data](https://www.nature.com/articles/s41597-023-02799-4) | `road-clear` vs. `road-blocked` semantic classes for logistical access. |
| 9 | **DisasTeller Multi-Agent LVLM (2024)** [arXiv:2411.01511](https://arxiv.org/pdf/2411.01511) | Cooperating micro-agent orchestration and advisory situation reporting. |
| 10 | **Drone vs. Satellite Disagreement (2025)** [arXiv:2505.08117](https://arxiv.org/pdf/2505.08117) | Calibration, uncertainty thresholding, and mandatory HITL approval gates. |
| 11 | **RL for Emergency Response Survey (2025)** [arXiv:2505.03979](https://arxiv.org/pdf/2505.03979) | Reinforcement learning for dispatch sequencing under resource constraints. |
| 12 | **Panda & Yadav (2025)** [arXiv:2506.22129](https://arxiv.org/pdf/2506.22129) | Damage-grade prediction tied to statutory relief-fund allocation formulas. |
| 13 | **Sun & Zhang (2020)** [Elsevier Transp Res](https://www.sciencedirect.com/science/article/abs/pii/S1361920920306428) | Interdependent infrastructure network resilience & repair crew prioritization. |
| 14 | **TEC Drone Study (Govt. of India)** [TEC Paper](https://www.tec.gov.in/pdf/Studypaper/study_paper_use_cases_of_drones_issued_copy.pdf) | National operational framework for disaster drone deployment. |
| 15 | **NDMA SACHET Early-Warning Platform** [sachet.ndma.gov.in](https://sachet.ndma.gov.in) | Oasis CAP v1.2 alert feed ingestion and multilingual warning protocols. |
| 16 | **DGCA Drone Rules 2021 & Digital Sky** [digitalsky.dgca.gov.in](https://digitalsky.dgca.gov.in) | 120m AGL ceiling, Green/Yellow/Red airspace verification, and NPNT gates. |

---

## Transition Roadmap: Simulation to Physical Fleet Deployment

| Subsystem | Simulation Implementation | Production SDMA Fleet Deployment |
| :--- | :--- | :--- |
| **Early Warning** | Polling simulated OASIS CAP v1.2 XML feeds. | Webhook subscription to NDMA’s production SACHET CAP Gateway and Survey of India (SoI) GIS boundaries. |
| **Edge Compute** | PyTorch models running on CPU/GPU. | TensorRT / ONNX compiled engines running on drone companion computers (NVIDIA Jetson Orin Nano). |
| **Monsoon SAR** | Simulated backscatter proxy. | Ingestion of real Sentinel-1 GRD / RISAT-1A SAR rasters via ISRO NRSC Bhoonidhi API for all-weather penetration. |
| **Drone Fleet** | MAVLink SITL flight simulator. | MAVSDK/ROS2 connected over telemetry serial (`/dev/ttyTHS1`); live Digital Sky REST API integration for dynamic NOTAMs and NPNT tokens; licensed RPIC maintaining RC manual override. |
| **Fund Disbursement** | MHA statutory schedules & explainable text ledger. | Integration with state land revenue databases (Bhu-Naksha Khasra numbers) and the Public Financial Management System (PFMS) for Direct Benefit Transfer (DBT) post-approval. |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
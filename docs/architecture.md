# System Architecture Specification: India Disaster Response Platform

## 1. Executive Summary & Core Philosophy
The platform is an autonomous, safety-critical, simulator-first AI decision-support system designed specifically for the operational, infrastructural, and regulatory environment of the Republic of India. 

### Fundamental Design Axioms:
1. **Zero-Hardware Decoupling**: All drone control logic speaks standardized MAVLink/MAVSDK schemas. The system operates identically whether executing against PX4 SITL/Gazebo in simulation or real companion computers (e.g. NVIDIA Jetson) on physical multicopters.
2. **Hierarchical Computer Vision**: Never run heavy segmentation or damage grading blindly. A lightweight Stage-1 triage classifier executes first on the edge to identify event type, routing to specialized Stage-2 severity/extent heads.
3. **No False Certainty (Sensor Disagreement Mitigated)**: Grounded in research showing drone vs. satellite label disagreement, predictions carry calibrated confidence metrics and an explicit `"uncertain / needs human review"` state.
4. **Separation of Statutory Law vs. Algorithmic Optimization**: Compensation numbers are **never learned by black-box models**. State Disaster Response Fund (SDRF) / National Disaster Response Fund (NDRF) statutory rates are deterministic, auditable lookup tables. Machine learning and optimization are strictly applied to **spatial prioritization, logistical sequencing, and anomaly detection**.
5. **Mandatory Human-in-the-Loop (HITL) Gate**: No irreversible action—disbursing funds, redirecting first responders, or deploying physical drone payloads—can occur autonomously in v1.
6. **Cryptographic Auditability & Privacy**: Every event is preserved in an append-only, SHA-256 hashed audit log. All human faces, license plates, and private home interiors are blurred prior to storage or external transmission per India's Digital Personal Data Protection (DPDP) Act.

---

## 2. High-Level System Architecture Diagram

```
                                  PUBLIC EARLY-WARNING FEEDS
                    ┌─────────────────────────────────────────────────────┐
                    │  NDMA SACHET (CAP v1.2 Feeds):                     │
                    │  - IMD (Cyclone / Severe Weather)                   │
                    │  - CWC (Central Water Commission - River Flood)    │
                    │  - INCOIS (Tsunami / Storm Surge Alerts)            │
                    │  - NRSC / Bhuvan (Satellite Flood Inundation Layers)│
                    │  - News / Crowdsourced Reports (Tagged Unverified)  │
                    └──────────────────────────┬──────────────────────────┘
                                               │ Polling / Push (Oasis CAP)
                                               ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ 1. LIVE MONITORING & GEOFENCING SERVICE             │
                    │    - Configurable Center-Point + Monitored Radius   │
                    │    - Spatial Polygon Intersection (PostGIS / Shapely)│
                    │    - Deduplication & Severity Escalation Engine     │
                    └──────────────────────────┬──────────────────────────┘
                                               │ TypedAlertEvent
                                               ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ 2. DISASTELLER MULTI-AGENT ORCHESTRATION LAYER      │
                    │    - Detection Agent                                │
                    │    - Assessment Agent                               │
                    │    - Resource & Fund Agent                          │
                    │    - Communication & Alert Agent                    │
                    │    - Multilingual Reporting Agent (SACHET 12-Lang)  │
                    └───────────┬───────────────────────────────┬─────────┘
                                │                               │
             Sensor Imagery     │                               │ Mission Request
             (Satellite/CCTV)   │                               │
                                ▼                               ▼
┌───────────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
│ 3. COMPUTER VISION & REASONING PIPELINE       │ │ 4. DRONE MISSION ABSTRACTION LAYER       │
│                                               │ │                                          │
│ ┌───────────────────────────────────────────┐ │ │ ┌──────────────────────────────────────┐ │
│ │ Stage 1: Fast Scene Classifier (Edge CNN) │ │ │ │ DGCA Drone Rules 2021 Safety Checker │ │
│ │ - MobileNetV3 / EfficientNet-Lite         │ │ │ │ - Green/Yellow/Red Airspace Zones    │ │
│ │ - Routes: Flood, Cyclone, Quake, Fire, etc│ │ │ │ - 120m (400 ft) AGL Ceiling Enforcer │ │
│ └─────────────────────┬─────────────────────┘ │ │ │ - NPNT (No Permission No Takeoff) Gate│ │
│                       │                       │ │ └──────────────────┬───────────────────┘ │
│                       ▼                       │ │                    │ Validated Mission   │
│ ┌───────────────────────────────────────────┐ │ │                    ▼                     │
│ │ Stage 2: Specialized Severity & Extent    │ │ │ ┌──────────────────────────────────────┐ │
│ │ - xBD 4-Level Ordinal Damage Grading      │ │ │ │ Hardware-Agnostic Mission Contract   │ │
│ │ - FloodNet Water / Building Segmentation  │ │ │ │ (Waypoints, Sensor Mode, RTH Limits) │ │
│ │ - RescueNet Road Passability / Blockages  │ │ │ └──────────────────┬───────────────────┘ │
│ │ - BRIGHT Multimodal Optical+SAR Fusion    │ │ │                    │                     │
│ └─────────────────────┬─────────────────────┘ │ │                    ▼                     │
│                       │                       │ │ ┌──────────────────────────────────────┐ │
│                       ▼                       │ │ │ PX4 SITL / MAVLink Flight Simulator  │ │
│ ┌───────────────────────────────────────────┐ │ │ │ - Standalone Mission Replay Engine   │ │
│ │ Stage 3: Multilingual VQA / VLM Reasoning │ │ │ │ - Simulated Live Aerial Camera Stream│ │
│ │ - Structured Narrative Incident Sit-Reps  │ │ │ └──────────────────┬───────────────────┘ │
│ │ - Natural Language Operator Q&A           │ │ │                    │                     │
│ │ - English, Hindi, Odia, Malayalam, etc.   │ │ │                    ▼                     │
│ └─────────────────────┬─────────────────────┘ │ │ ┌──────────────────────────────────────┐ │
│                       │                       │ │ │ On-Scene Re-Analysis Loop            │ │
│                       ▼                       │ │ │ - Oblique Multi-Altitude Inspection  │ │
│ ┌───────────────────────────────────────────┐ │ │ │ - Constrained Action Enum Decision   │ │
│ │ Model Registry, Rollback & Calibration    │ │ │ └──────────────────┬───────────────────┘ │
│ │ - Expected Calibration Error (ECE) Minim. │ │ └────────────────────┼─────────────────────┘
│ │ - 'Uncertain / Review' Routing (< 70% Conf)│                       │
│ └─────────────────────┬─────────────────────┘                        │
└───────────────────────┼──────────────────────────────────────────────┘
                        │ Quantitative Damage Metrics & Road Network Status
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. IMPACT & NEEDS ENGINE                                                               │
│    - Overlay: WorldPop Population Grid + OpenStreetMap Critical Infrastructure          │
│    - Infrastructure Vulnerability: Hospitals, Substations, Bridges, Water Treatment    │
│    - Needs Vector: [Medical, Food/Water, Shelter, Evacuation, Structural] (0.0 to 1.0) │
└───────────────────────────────────────┬────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. DEFENSIVE SDRF / NDRF FUND-ALLOCATION ENGINE                                        │
│    - Statutory Compensation Schedules (MHA Official Rates for Pucca / Kutcha / Crop)   │
│    - Explainable Breakdown: Plain-language, audited line-item derivation of funds       │
│    - Discrepancy & Fraud Detector: Flags divergence between imagery and self-claims    │
└───────────────────────────────────────┬────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 7. SAFETY-CRITICAL COMMAND CENTER & HUMAN-IN-THE-LOOP GATE                             │
│    - Hard Gate: Operator Signature required for flight launch, fund release, crew task │
│    - DPDP Automated Redaction: Blurring of faces, license plates, private interiors    │
│    - Cryptographic Audit Log: Append-only SHA-256 chained transaction trail             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Breakdown

### Component 1: Live Monitoring & Geofencing Ingestion
- **Protocol**: Consumes Oasis CAP (Common Alerting Protocol) v1.2 XML/JSON.
- **Feeds**: NDMA SACHET, IMD (India Meteorological Dept), CWC (Central Water Commission), INCOIS (Indian National Centre for Ocean Information Services).
- **Geofencing**: Given a latitude, longitude, and radius (e.g. 75 km), computes spherical distance and polygon containment. Emits typed `InternalAlertEvent` only when an alert intersects the operational theater.

### Component 2: Multi-Stage Computer Vision & Advisory Reasoning
- **Stage 1 (Triage)**: MobileNetV3 / EfficientNet-Lite classifier trained on AIDER + India-augmented imagery. Runs in <50ms per frame.
- **Stage 2 (Disaster-Specific Assessment)**:
  - *Earthquake / Structural*: 4-level ordinal damage grade (`0: No Damage`, `1: Minor`, `2: Major`, `3: Destroyed`) trained on xBD.
  - *Flood*: Semantic segmentation of water extent and flooded building masks trained on FloodNet.
  - *Cyclone / Landslide*: Debris density and road passability (`road-clear` vs `road-blocked`) trained on RescueNet.
  - *Monsoon Robustness*: `MultimodalSARFusion` interface fusing optical RGB with Sentinel-1/RISAT SAR backscatter to penetrate cloud cover.
- **Stage 3 (Advisory VLM & Multilingual Reporting)**:
  - DisasTeller-style Vision-Language reasoning generating structured operational briefs in 7 primary Indian disaster languages (English, Hindi, Odia, Malayalam, Bengali, Tamil, Telugu).
  - Strictly advisory: text outputs cannot override numerical CV damage indices.
- **Calibration & "No False Certainty" Gate**:
  - Platt / Temperature scaling ensures probabilities reflect true accuracy.
  - Any classification with confidence $< 0.70$ or high entropy ($> 1.25$) is routed to `uncertain_needs_review`.

### Component 3: Drone Mission Abstraction & DGCA Compliance
- **Contract**: Decoupled Pydantic JSON schema specifying survey bounding box, grid spacing, altitude AGL, camera trigger intervals, and fail-safes.
- **Regulatory Gate (Drone Rules 2021)**:
  - Enforces altitude ceiling $\le 120\text{ m}$ (400 ft) AGL.
  - Queries Digital Sky airspace maps: auto-rejects missions intersecting Red Zones (e.g. military/airports) unless accompanied by a digitally signed MoCA authorization token.
  - Enforces NPNT (No Permission No Takeoff) compliance and maintains Remote Pilot in Command (RPIC) emergency override link.
- **Simulator**: MAVLink-speaking flight harness compatible with PX4 SITL and Gazebo. Simulates battery draw, telemetry streaming, and automated return-to-home.
- **On-Scene Agent**: Inspects real-time video frames and selects exclusively from an approved action enum: `SURVEY_GRID`, `WIDEN_SEARCH`, `MARK_HOTSPOT`, `REQUEST_SUPPLY_DROP`, `REQUEST_HUMAN_REVIEW`, `RETURN_TO_HOME`.

### Component 4: Impact, Needs & Fund Allocation Engine
- **Impact & Needs Engine**: Combines damage polygons with OpenStreetMap infrastructure footprints and WorldPop population grids. Computes urgent needs vectors across medical, shelter, water/food, evacuation, and structural shoring.
- **Fund Allocation Analytics**:
  - Applies official Ministry of Home Affairs SDRF/NDRF rates (e.g. ₹1,20,000 for fully destroyed RCC pucca house, ₹80,000 for kutcha house, ₹8,500/ha for rainfed crop loss).
  - Outputs an itemized, explainable ledger showing exactly which physical cues generated the financial estimates.
  - Flags self-reported claim anomalies where claims exceed imagery-derived estimates by $> 50\%$.

### Component 5: Command Center, HITL Gate & DPDP Privacy
- **Human-in-the-Loop Gateway**: Every actionable proposal (launching a mission, releasing relief funds, tasking emergency responders) enters a pending queue in the Command Center. Requires affirmative officer confirmation with role-based cryptographic signing.
- **DPDP Automated Redaction**: An embedded pre-processing filter detects human faces and vehicle license plates, applying Gaussian blurring before storage or distribution outside the tactical ops room.
- **Cryptographic Audit Log**: Every system event (ingestion, prediction, human approval, mission launch, fund release) is recorded with timestamp, actor ID, payload hash, and previous block hash in an append-only ledger (`audit_trail.jsonl`).

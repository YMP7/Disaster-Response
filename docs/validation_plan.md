# Comprehensive Zero-Hardware Validation Plan

## 1. Validation Strategy & Philosophy
Because physical drone hardware cannot be deployed during development, this platform employs a rigorous **Three-Tiered Zero-Hardware Validation Framework**:

```
                 ▲
                / \
               /   \      Tier 3: Full SITL Closed-Loop Simulation
              / Tier\     - PX4 SITL / MAVLink flight mission replay
             /   3   \    - Real-time on-scene video streaming & re-classification
            /─────────\   - HITL command gate & audit trail enforcement
           /   Tier    \
          /     2       \  Tier 2: Real India Held-Out Benchmark Testing
         /───────────────\ - Kerala 2018 Floods, Odisha 2019 Cyclone Fani, Chamoli
        /      Tier       \- Stress-test against domain shift & turbidity
       /        1          \
      /─────────────────────\ Tier 1: Algorithmic & Regulatory Unit Invariants
     /                       \- DGCA 120m ceiling, SDRF rates, DPDP redaction, ECE
```

---

## 2. Tier 1: Algorithmic Invariants & Regulatory Boundaries

### 1. Aviation Compliance (DGCA Drone Rules 2021)
- **Ceiling Enforcement**: Inject missions with altitudes $> 120\text{ m}$ (400 ft) AGL; verify that `DGCAComplianceChecker` throws a hard violation.
- **Red Zone Proximity**: Place waypoints within 3 km of defense and airport perimeters; verify rejection with zero tolerance.
- **NPNT Hardware Gate**: Attempt mission generation with empty/invalid NPNT tokens; verify takeoff prevention.

### 2. Statutory Relief Mathematics (SDRF/NDRF Norms)
- **Deterministic Valuation**: Verify that compensation numbers match published Ministry of Home Affairs schedules:
  - Pucca destroyed: ₹1,20,000
  - Kutcha destroyed: ₹80,000
  - Rainfed crop loss: ₹8,500/ha
- **Zero Black-Box Pricing**: Ensure no currency value is predicted via unconstrained regression weights.

### 3. Personal Privacy (DPDP Act Compliance)
- **PII Scrubbing**: Feed imagery with human faces and automobile license plates; verify Gaussian blur application prior to storage or external API serialization.

### 4. Cryptographic Non-Repudiation (Audit Trail)
- **Hash Chain Verification**: Audit `audit_trail.jsonl` block-by-block. Verify that modifying or deleting a previous block immediately breaks the SHA-256 chain and fails integrity audit.

---

## 3. Tier 2: Real India Held-Out Benchmark Suite

To prevent overfitting to synthetic imagery, models are evaluated against curated held-out real Indian disaster fixtures:

1. **Kerala 2018 Floods Benchmark**:
   - *Domain Challenges*: High-silt brownish water, terracotta Mangalore tile roofs, heavy monsoon rain streaks.
   - *Target Metric*: Water extent estimation within $\pm 7\%$ IoU of ground truth.
2. **Odisha 2019 Cyclone Fani Benchmark**:
   - *Domain Challenges*: Twisted tin/GI sheets, fallen coconut palms, intense tropical haze.
   - *Target Metric*: Structural damage classification accuracy $> 85\%$ on unroofed buildings.
3. **Uttarakhand 2021 Chamoli Flash Flood Benchmark**:
   - *Domain Challenges*: Steep Himalayan mountain gorges, gray slurry sediment, cut transportation corridors.
   - *Target Metric*: 100% detection of severed highway corridors (`ROAD_BLOCKED`).

---

## 4. Tier 3: Full SITL Mission Replay & On-Scene Loop

Simulates the end-to-end incident lifecycle:
1. **SACHET Ingestion**: Feed live or historical OASIS CAP alert for coastal Odisha.
2. **Geofence Check**: Compute spatial intersection with the 80 km operational radius around Cuttack.
3. **Mission Generation**: Plan lawnmower survey grid at 60m AGL. Verify DGCA clearance.
4. **PX4 SITL Flight Execution**: Drone takes off in simulation, streaming telemetry (`GLOBAL_POSITION_INT`, battery drawdown) and synthesized camera frames at every waypoint.
5. **On-Scene Re-Analysis**: Companion computer agent re-classifies incoming video, detects submerged buildings, and triggers a constrained `REQUEST_SUPPLY_DROP` action.
6. **Command Center Gate**: Action halts in pending queue. Only affirmative human operator review allows execution and appends cryptographic audit record.

# Disaster Response AI — Dataset Placement & Download Guide

This document specifies the dataset sources, download links, expected directory layouts, and ingestion procedures for training and evaluating the vision models.

---

## Quick Reference Summary

| Dataset | Primary Role | Domain & Modality | Approx Size | Download Access & Verified Links | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AIDER** | Stage-1 Disaster Scene Triage | UAV / Drone RGB (oblique) | **~2.5 GB** | [Kaggle Dataset (clguo1/aiderdata)](https://www.kaggle.com/datasets/clguo1/aiderdata) | **Highest (Start here)** |
| **FloodNet** | Stage-2 Flood Severity & VQA | UAV RGB (high-res oblique) | **~5.2 GB** | [Dropbox Direct Archive](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=5&dl=0) | **High** |
| **RescueNet** | Road Passability & Structural Damage | UAV RGB (Post-Hurricane Ian) | **~10.4 GB** | [Dropbox Direct Archive](https://www.dropbox.com/scl/fo/ntgeyhxe2mzd2wuh7he7x/AHJ-cNzQL-Eu04HS6bvBgcw?rlkey=6vxiaqve9gp6vzvzh3t5mz0vv&e=6&dl=0) | **Medium** |
| **xBD (xView2)** | Building Damage Multi-Hazard Scale | Satellite RGB (pre + post disaster) | **~35 GB** | Free Registration ([xview2.org](https://xview2.org)) | **Optional / Advanced** |
| **ISRO Bhuvan** | India Ground-Truth Flood Vectors | Satellite Optical + RISAT SAR GeoTIFF | **Variable** | Open Data ([bhuvan.nrsc.gov.in](https://bhuvan-app1.nrsc.gov.in/disaster/)) | **India Field Ops** |

---

## 1. AIDER (Aerial Image Database for Emergency Response)
*Recommended First Download: Smallest footprint, directly maps to Stage-1 edge classification.*

- **Paper / Grounding**: Kyrkou & Theocharides (CVPRW 2019, IEEE JSTARS EmergencyNet 2021)
- **Primary Download Link (Kaggle)**: [https://www.kaggle.com/datasets/clguo1/aiderdata](https://www.kaggle.com/datasets/clguo1/aiderdata)
- **GitHub Reference Repo**: [https://github.com/philipposkyrkou/AIDER](https://github.com/philipposkyrkou/AIDER)
- **Classes**: `Fire/Smoke`, `Flood`, `Collapsed Building`, `Traffic Accident`, `Normal`.
- **Expected Directory Structure**:
  ```text
  data/
  └── aider/
      ├── fire/
      │   ├── fire_0001.jpg
      │   └── ...
      ├── flood/
      │   ├── flood_0001.jpg
      │   └── ...
      ├── collapsed_building/
      │   ├── collapse_0001.jpg
      │   └── ...
      └── normal/
          ├── normal_0001.jpg
          └── ...
  ```

---

## 2. FloodNet (UAV Flood Scene Understanding & VQA)
*High-resolution oblique UAV imagery for flooded structure counting and water segmentation.*

- **Paper / Grounding**: Rahnemoonfar et al. (IEEE Access 2021)
- **Primary Download Link (Dropbox)**: [FloodNet Dropbox Archive](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=5&dl=0)
- **GitHub Reference Repo**: [https://github.com/Bina-Lab/FloodNet-Supervised_v1.0](https://github.com/Bina-Lab/FloodNet-Supervised_v1.0)
- **Expected Directory Structure**:
  ```text
  data/
  └── floodnet/
      ├── train/
      │   ├── train-org-img/
      │   └── train-label-img/
      ├── val/
      │   ├── val-org-img/
      │   └── val-label-img/
      └── Questions/
          ├── FloodNet_v1.0_VQA_Train.json
          └── FloodNet_v1.0_VQA_Val.json
  ```

---

## 3. RescueNet (UAV Natural Disaster Assessment)
*Essential for Road Passability (Road-Clear vs Road-Blocked) and roof damage segmentation.*

- **Paper / Grounding**: Bina-Lab (IEEE TGRS / CVPRW)
- **Primary Download Link (Dropbox)**: [RescueNet Dropbox Archive](https://www.dropbox.com/scl/fo/ntgeyhxe2mzd2wuh7he7x/AHJ-cNzQL-Eu04HS6bvBgcw?rlkey=6vxiaqve9gp6vzvzh3t5mz0vv&e=6&dl=0)
- **GitHub Reference Repo**: [https://github.com/Bina-Lab/RescueNet](https://github.com/Bina-Lab/RescueNet)
- **Expected Directory Structure**:
  ```text
  data/
  └── rescuenet/
      ├── train/
      │   ├── train-org-img/
      │   └── train-label-img/
      └── val/
          ├── val-org-img/
          └── val-label-img/
  ```

---

## 4. xBD (xView2 Building Damage Assessment)
*Satellite-scale 4-tier damage classification across 19 global disaster events.*

- **Official Portal**: [https://xview2.org/dataset](https://xview2.org/dataset) (Sign in with free registration)
- **Expected Directory Structure**:
  ```text
  data/
  └── xbd/
      ├── train/
      │   ├── images/
      │   │   ├── hurricane-florence_00000001_pre_disaster.png
      │   │   └── hurricane-florence_00000001_post_disaster.png
      │   └── labels/
      │       ├── hurricane-florence_00000001_pre_disaster.json
      │       └── hurricane-florence_00000001_post_disaster.json
      └── test/
          ├── images/
          └── labels/
  ```

---

## 5. ISRO Bhuvan Indian Disaster Services
*India-specific disaster footprints, flood inundation vector polygons, and cloud-penetrating RISAT-1A SAR layers.*

- **Official Portal**: [https://bhuvan-app1.nrsc.gov.in/disaster/](https://bhuvan-app1.nrsc.gov.in/disaster/)
- **Expected Directory Structure**:
  ```text
  data/
  └── india_bhuvan/
      ├── shapefiles/
      │   ├── Assam_Flood_2024_Inundation.shp
      │   └── Odisha_Cyclone_Dana_2024.shp
      └── sar_geotiff/
          └── RISAT1A_CBand_Assam_Flood.tif
  ```

---

## Ingestion Workflows

Once you extract any of the datasets into `data/<dataset_name>/`:
Use `data_pipeline/ingestion.py` adapters to parse and normalize the raw files into `UnifiedDisasterAnnotation` dataclass instances.

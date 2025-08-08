# 🏗️ Scan-to-EDTs: Scan to Energy Digital Twin

This repository provides an end-to-end framework for converting building scan data into rich energy simulation models and Digital Twins. It leverages geometric assessment, semantic enrichment, and data-driven modeling to enable **energy-efficient building management and forecasting**.

---

## 🚀 Workflow Overview

### 🔍 1. Input & Preprocessing
- Input **Point Cloud** & Classification
  - tested via Point Transformer v3, Pointcept, SegmentorV2
  - Classes  0 floor, 1 ceiling, 2 wall, 3 column, 4 door, 5 window

### 🛠️ 2. 2D and 3D Modelling 
- **Scan-to-BIM** (Solid Model) [Roman et al., 2024]
- **Scan-to-BEM** (devices detection)
  - Classes 6 ligth, 7 radiator, 8 hvac devices    
- **Geometric Assessment**
  - **Solid Model**
  - **Topologic B-REP Model**

### 🧠 3. Semantic Enrichment
- Thickness  
- Materials  
- Dimensions  

### 🧾 4. Rules and Parameters
  Apply **Information Loading Dictionaries (ILDs)**:
- Setpoints (Tmin, Tmax)  
- Devices & Rules (e.g., European Regulations)

➡️ Output **Building Energy Model (BEM) B-REP based** for simulation

### 🧱 5. Model Transformation
Convert to:
- IFC Model  
- gbXML  
- Topologic Models (Volumetric & Thematic Surface Models)

➡️ Export using **EPJSON** via parser/transformer/writer tools

### ⚙️ 6. Simulation & Analysis
🎯 Run with **EnergyPlus**  
Use **Real-Time IoT Data** & **ML/DL algorithms** for:

- 📈 **Digital Twin Level 1**: Real-Time Monitoring  
- 🧪 **Digital Twin Level 2**: Simulation Scenarios  
- 🔮 **Digital Twin Level 3**: Forecasting and Scenario Testing

---

| Module              | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `scan2bim/`         | Scripts for converting classified point clouds into solid BIM geometry  |
| `scan2edts/`        | Tools for generating topological B-REP models                |
| `semantic_loader/`  | Load ILDs and attach semantic/typological metadata           |
| `model_ifc_converter/`  | Utilities for IFC model conversion     |
| `model_gbxml_converter/`  | Utilities for gbXML model conversion     |
| `model_epjson_converter/`  | Utilities for epJSON generator and model conversion     |
| `energy_simulator/` | Integration with EnergyPlus (EPJSON-based simulation)        |
| `iot_interface/`    | Real-time ingestion and formatting of IoT sensor data        |
| `digital_twin_module/`     | Modules for monitoring, simulation, and ML-based forecasting |


## 📊 Output Examples

- Geometric Models for analysis & compliance checks
  - wkt primary output 
  - BIM/IFC
  - BEM B-REP
  - gbXML
  - epJSON
- Energy Simulation Graphs showing heating/cooling loads  
- IoT Monitoring Dashboards for real-time insights  

---

## 🔧 Requirements

- Python 3.8 +  
- EnergyPlus  
- OpenCascade
- Topologic  
- IFCOpenShell    
- PyTorch (TorchGeometric)
  - soon .yml file releasing

---

## 💡 Use Cases

- Retrofitting analysis  
- HVAC optimization  
- Smart building simulation  
- Building performance benchmarking  

---
# 🏗️ Scan-to-EDTs Pipeline

Transform a classified point cloud into a **semantically rich**, **simulation-ready**, and **BIM-compatible** building model.

---

## 🔹 1. 🛰️ Classify the Point Cloud

Segment your 3D scan and assign semantic labels:
🧱 wall | 🪟 window | 🚪 door | 🧼 ceiling | 🪜 floor | 📏 column | ❓ unclassified

---

## 🔹 2. 🗂️ Map Classes in a JSON

Reference raw class IDs to standardized labels:

```
{ "0": "floor", "1": ceiling, "2" "wall","3": "column", "4": "door", "5": "window", "99": "unclassified" }
```

---

## 🔹 3. 🧭 Generate Topological Graph

Run `graph_generator` to compute spatial relationships:
➡️ adjacency | 🔗 connectivity | 🧩 grouping | 🏛️ hierarchy

---

## 🔹 4. 🧊 Create EDT Structure

Run: `step_01_scan_to_edts`
🔁 Converts the point cloud into a solid volume with:

* Topological Solid Model (TSM)
* Thematic Surface Model (TSurf)

---

## 🔹 5. 🧱 Generate B-Rep Geometry

Run: `step_02_scan_to_edts`
🔁 Outputs clean boundary surfaces with full topology and semantics.

---

## 🔹 6. 🎨 Assign Materials and Layers

Run: `step_03_assign_material`
📄 Output: `_with_materials.csv`
Convert to `.epJSON` using `epJSON_parser` for EnergyPlus compatibility.

---

## 🔹 7. 🔥 Run Energy Simulation & Export gbXML

Run: `step_04_energy_simulation_uep`
📄 Outputs:

* `<thermal_zone>_temperatures.csv`
* `<thermal_zone>_boundaries.csv`

---

## 🔹 8. 🏢 Generate Digital Twin Data

🧾 Use `edts_module` to:

* Generate a valid Digital Twin (DT)
* Add sensors, lights, radiators
* Inspect, analyze, and visualize results

---

## 🔹 9. 🏗️ Export to IFC (Optional)

Use: `IFC_parser_transformer_writer`
➡️ Export full model to `.IFC` for BIM tools (e.g., Revit, BIMcollab)


## 📎 References

- [EnergyPlus](https://energyplus.net/)  
- [Topologic](https://topologic.app/)  
- [IFCOpenShell](https://ifcopenshell.org/)  

---

## 👨‍💻 Contributing

//


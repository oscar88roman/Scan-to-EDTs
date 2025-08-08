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

# 🏗️ End-to-End Building Model Generation Pipeline
Turn a classified point cloud into a semantically rich, simulation-ready, and BIM-compatible building model.

🔹 1. 🛰️ Classify Your Point Cloud
> Segment your 3D scan and assign labels such as:

🧱 wall | 🪟 window | 🚪 door | 🧼 ceiling | 🛗 floor | 🪜 column | ❓ unclassified

🔹 2. 🗂️ Reference Classes in a JSON
> Map raw class IDs to standard building types:

{ "1": "wall", "2": "floor", "3": "window", "99": "unclassified" }

🔹 3. 🧭 Generate Topological Graph
> Run graph_generator to compute spatial relationships:

➡️ adjacency | connectivity | grouping

🔹 4. 📦 Create EDT Structure (Solid Model + Thematic Surface Model)
> Run: step_01_scan_to_edts

🔁 Converts point cloud into solid EDT-based volume.

🔹 5. 🧱 Generate B-Rep Geometry
> Run: step_02_scan_to_edts

🔁 Outputs clean boundary surfaces with topology and semantics.

🔹 6. 🧪 Assign Materials and Layers
> Run: step_03_assign_material

📄 Output: <filename>_with_materials.csv
> Use epJSON_parser to convert into .epJSON for EnergyPlus.

🔹 7. 🔥 Run Simulation & Export gbXML
> Run: step_04_energy_simulation_uep

> 📄 Output: <filename>_<thermal_zone>_temperatures.csv
> 📄 Output: <filename>_<thermal_zone>_boundaries.csv

🔹8. 🏢 DT module
🧾 Generates valid DT with sensors/ligths/radiators positions
> Use: edts_module to inspect results, use data for analysis, visualise results.

🔹 9. 🏢 Export to IFC (Optional)
> Use: IFC_parser_transformer_writer

➡️ Export the full model to .IFC for BIM software (Revit, BIMcollab, etc.)


## 📎 References

- [EnergyPlus](https://energyplus.net/)  
- [Topologic](https://topologic.app/)  
- [IFCOpenShell](https://ifcopenshell.org/)  

---

## 👨‍💻 Contributing

//


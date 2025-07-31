# 🏗️ Scan-to-BIM to Energy Digital Twin

This repository provides an end-to-end framework for converting building scan data into rich energy simulation models and Digital Twins. It leverages geometric assessment, semantic enrichment, and data-driven modeling to enable **energy-efficient building management and forecasting**.

---

## 🚀 Workflow Overview

### 🔍 1. Input & Preprocessing
- Input **Point Cloud** & Classification via Point Transformer v3 and Pointcept
- **Scan-to-BIM** (Solid Model) [Roman et al., 2024]
- **Geometric Assessment**
- **Solid Model**
- **Topologic B-REP Model**

### 🧠 2. Semantic Enrichment
Apply **Information Loading Dictionaries (ILDs)**:
- Thickness  
- Materials  
- Dimensions  
- Setpoints (Tmin, Tmax)  
- Devices & Rules (e.g., European Regulations)

➡️ Outputs a **Building Energy Model (BEM)** for simulation

### 🧱 3. Model Transformation
Convert to:
- IFC Model  
- gbXML  
- Topologic Models (Volumetric & Thematic Surface Models)

➡️ Export using **EPJSON** via parser/transformer/writer tools

### ⚙️ 4. Simulation & Analysis
Run with **EnergyPlus**  
Use **Real-Time IoT Data** & **ML/DL algorithms** for:

- 📈 **Digital Twin Level 1**: Real-Time Monitoring  
- 🧪 **Digital Twin Level 2**: Simulation Scenarios  
- 🔮 **Digital Twin Level 3**: Forecasting and Scenario Testing

---

| Module              | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `scan2bim/`         | Scripts for converting point clouds into solid BIM geometry  |
| `scan2edts/`        | Tools for generating topological B-REP models                |
| `semantic_loader/`  | Load ILDs and attach semantic/typological metadata           |
| `model_converter/`  | Utilities for IFC, gbXML, and Topologic model conversion     |
| `energy_simulator/` | Integration with EnergyPlus (EPJSON-based simulation)        |
| `iot_interface/`    | Real-time ingestion and formatting of IoT sensor data        |
| `digital_twin/`     | Modules for monitoring, simulation, and ML-based forecasting |


## 📊 Output Examples

- Geometric Models for analysis & compliance checks  
- Energy Simulation Graphs showing heating/cooling loads  
- IoT Monitoring Dashboards for real-time insights  

---

## 🔧 Requirements

- Python 3.8+  
- EnergyPlus  
- OpenCascade / Topologic  
- IFCOpenShell  
- NumPy, Pandas, Scikit-learn  
- TensorFlow / PyTorch (optional, for ML modules)  

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
Segment your 3D scan and assign labels such as:

🧱 wall | 🪟 window | 🚪 door | 🧼 ceiling | 🛗 floor | 🪜 column | ❓ unclassified

🔹 2. 🗂️ Reference Classes in a JSON
Map raw class IDs to standard building types:
{ "1": "wall", "2": "floor", "3": "window", "99": "unclassified" }

🔹 3. 🧭 Generate Topological Graph
Run graph_generator to compute spatial relationships:
➡️ adjacency | connectivity | grouping

🔹 4. 📦 Create EDT Structure (Solid Model)
Run: step_01_scan_to_edts
🔁 Converts point cloud into solid EDT-based volume.

🔹 5. 🧱 Generate B-Rep Geometry
Run: step_02_scan_to_edts
🔁 Outputs clean boundary surfaces with topology and semantics.

🔹 6. 🧪 Assign Materials and Layers
Run: step_03_assign_material
📄 Output: <filename>_with_materials.csv
Use epJSON_parser to convert into .epJSON for EnergyPlus.

🔹 7. 🔥 Run Simulation & Export gbXML
Run: step_04_energy_simulation_uep
🧾 Generates valid gbXML
Use: gbXML_parser_transformer_writer to inspect/edit.

🔹 8. 🏢 Export to IFC (Optional)
Use: IFC_parser_transformer_writer
➡️ Export the full model to .IFC for BIM software (Revit, BIMcollab, etc.)


## 📎 References

- [EnergyPlus](https://energyplus.net/)  
- [Topologic](https://topologic.app/)  
- [IFCOpenShell](https://ifcopenshell.org/)  

---

## 👨‍💻 Contributing

PRs are welcome! For major changes, please open an issue first to discuss what you would like to change.


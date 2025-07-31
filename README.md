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
🏗️ End-to-End Building Model Generation Pipeline
This pipeline converts a classified point cloud into a semantically rich, simulation-ready, and BIM-compatible building model.

🔹 1. 🛰️ Classify your point cloud
Segment your 3D scan using your preferred tool and assign labels to elements like:
🧱 walls | 🪟 windows | 🚪 doors | 🧼 ceilings | 🛗 floors | 🪜 columns | ❓ unclassified

🔹 2. 🗂️ Reference classes in a JSON
Define a mapping between your point cloud labels and building elements:

{ "1": "wall", "2": "floor", "3": "window", "99": "unclassified" }

🔹 3. 🧭 Create the topological graph
Use the graph_generator script to build spatial relationships (adjacency, connectivity) between components.

🔹 4. 📦 From scan → EDT structure (solid model)
Run step_01_scan_to_edts to convert the labeled point cloud into a volume-based solid representation.

🔹 5. 🧱 Generate the B-Rep geometry
Run step_02_scan_to_edts to extract explicit boundary surfaces from the EDT structure.
Result: a clean, watertight geometry.

🔹 6. 🧪 Assign material and construction layers
Run step_03_assign_material to attach materials and layered construction properties to each element.
⬇️ Output: <filename>_with_materials.csv
Then use the epJSON_parser to generate EnergyPlus input.

🔹 7. 🔥 Simulate & export gbXML
Run step_04_energy_simulation_uep to simulate building performance and generate a valid gbXML model.
Use gbXML_parser_transformer_writer to inspect or transform this export.

🔹 8. 🏢 Export to IFC (optional)
Use IFC_parser_transformer_writer to convert your enriched model into an IFC file for use in BIM software (Revit, BIMcollab, etc.).


## 📎 References

- [EnergyPlus](https://energyplus.net/)  
- [Topologic](https://topologic.app/)  
- [IFCOpenShell](https://ifcopenshell.org/)  

---

## 👨‍💻 Contributing

PRs are welcome! For major changes, please open an issue first to discuss what you would like to change.


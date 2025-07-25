# 🏗️ Digital Twin Geometry Pipeline for BIM & Energy Assessment

<img src="./docs/f9da4944-e6e0-4187-8498-26e9d1bbe01a.png" width="100%" />

## 📦 Overview

This repository provides a full Python pipeline for creating **semantically rich IFC files** from pre-processed geometric models (e.g., CityJSON, GeoJSON, point clouds), enriched with thickness, height, surface type and WKT-based geometry. It's designed to support:

- 🧱 **Wall, floor, ceiling, door, window, and space generation**
- 📐 **3D geometric reconstruction and spatial placement**
- ♻️ **IFC4 export (via `ifcopenshell`)**
- 🔌 **Integration with BEM tools like EnergyPlus or EPJSON**
- 🧠 **ML-ready output for digital twin simulation & prediction**

---

## 🔧 Features

- ✅ Automatic **thickness inference** per surface type (`external_wall`, `party_wall`, etc.)
- ✅ Dynamic **wall height extraction** from 3D geometry (`z_min`, `z_max`)
- ✅ Geometry parsing via **WKT/`shapely`**
- ✅ IFC creation using **`ifcopenshell`** with spatial hierarchy: Project → Building → Storey → Elements
- ✅ Support for:
  - 🧱 `IfcWallStandardCase`
  - 🪟 `IfcWindow`, 🚪 `IfcDoor`
  - 🧭 `IfcSpace`
  - 🏗️ `IfcSlab`, `IfcCovering` (floor, ceiling)
- ✅ Clean object placement, local positioning, and profile-based extrusion

---

## 📂 Workflow

<img src="./docs/762a6699-db65-49fb-8243-d644cc57488f.png" width="100%" />

### 📉 Input

- `df_building_`: Geometry and thematic surfaces (walls, floors, ceilings)
- `df_door`, `df_window`, `df_room`: Optional semantic objects
- `surface_vertices`: 3D polygons defining geometry

### 🔄 Process

1. **Parse & clean geometry** (WKT, surface triangulation, z-min/max)
2. **Infer thickness & height**
3. **Generate wall/floor/ceiling/window/door entities**
4. **Attach to building storey**
5. **Export IFC4-compliant model**

---

## 🚀 Quickstart

```bash
pip install pandas shapely ifcopenshell numpy

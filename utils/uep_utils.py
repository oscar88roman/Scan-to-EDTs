import os
import re
from pathlib import Path

import pandas as pd
import numpy as np

from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon

import trimesh
from trimesh import Trimesh
from trimesh.util import concatenate
from trimesh import Trimesh
from trimesh.util import concatenate
from shapely.geometry import Polygon
import numpy as np

from collections import defaultdict
import json

# def extract_vertices(wkt_string):
#     try:
#         geom = wkt.loads(wkt_string)
#         if isinstance(geom, Polygon):
#             coords = list(geom.exterior.coords)
#         elif isinstance(geom, MultiPolygon):
#             # Grab only the largest polygon
#             largest = max(geom.geoms, key=lambda p: p.area)
#             coords = list(largest.exterior.coords)
#         else:
#             return []  # Unknown geometry

#         # Ensure 3D
#         vertices = [(float(x), float(y), float(z)) for x, y, z in coords if z is not None]
#         if len(vertices) >= 3:
#             return [vertices]  # Nested to fit your current format
#         else:
#             return []
#     except Exception as e:
#         print("⚠️ WKT parse error:", e)
#         return []
    

def extract_vertices(wkt_str):

    if not isinstance(wkt_str, str):
        return []

    match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", wkt_str)
    
    if len(match) % 3 != 0:
        return []  # Not divisible into triplets

    try:
        coords = [tuple(map(float, match[i:i+3])) for i in range(0, len(match), 3)]
        return coords
    except Exception as e:
        print(f"[ERROR] Failed to extract from: {wkt_str}")
        return []


def is_valid_surface(coords):
    if not coords or len(coords) < 3:
        return False
    coords = list({(round(x, 6), round(y, 6), round(z, 6)) for x, y, z in coords})
    if len(coords) < 3:
        return False

    def triangle_area(p1, p2, p3):
        a = [p2[i] - p1[i] for i in range(3)]
        b = [p3[i] - p1[i] for i in range(3)]
        cross = [a[1]*b[2] - a[2]*b[1],
                 a[2]*b[0] - a[0]*b[2],
                 a[0]*b[1] - a[1]*b[0]]
        return 0.5 * (sum(c**2 for c in cross) ** 0.5)

    try:
        area = triangle_area(*coords[:3])
        return area > 0.01
    except Exception as e:
        print("Area check failed:", e)
        return False


def get_surface_type(raw_type, row=None, fenestration_elements=None):
    mapping = {
        "external": "Wall",
        "external_with_hole": "Wall",
        "partly_wall": "Wall",
        "partly_wall_with_hole": "Wall",
        "wall": "Wall",
        "ceiling": "Roof",
        "roof": "Roof",
        "floor": "Floor"
        # 'window' and 'door' are intentionally excluded
    }

    surface_type = mapping.get(raw_type.lower())

    if surface_type is None and raw_type.lower() in ["door", "window"]:
        if fenestration_elements is not None and row is not None:
            fenestration_elements.append(row)
        return None

    return surface_type


def check_watertightness_requirement(room_id_to_surfaces):
    required_types = {"Floor", "Roof", "Wall"}
    
    valid_rooms = {}
    invalid_rooms = {}

    for room_id, surfaces in room_id_to_surfaces.items():
        vertices = []
        faces = []
        surface_types = set()
        has_invalid_surface = False
        vertex_index = 0

        for _, row in surfaces.iterrows():
            verts = row["vertices"]

            if not verts or len(verts) < 3:
                print(f"[ERROR] Room {room_id}, surface {row['element_id']} has too few vertices.")
                has_invalid_surface = True
                continue

            # Add surface type
            surface_type = row.get("thematic_surface") or row.get("element_type")
            if surface_type:
                surface_types.add(surface_type)

            n = len(verts)
            indices = list(range(vertex_index, vertex_index + n))
            vertices.extend(verts)
            faces.append(indices)
            vertex_index += n

        # Skip room if invalid surfaces
        if has_invalid_surface:
            invalid_rooms[room_id] = surfaces
            continue

        # Ensure required types
        simplified_types = {t.lower() for t in surface_types}
        missing = {t.lower() for t in required_types} - simplified_types
        if missing:
            print(f"[WARNING] Room {room_id} missing surface types: {missing}")
            invalid_rooms[room_id] = surfaces
            continue

        try:
            mesh = Trimesh(vertices=np.array(vertices), faces=faces, process=True)
            if mesh.is_watertight:
                valid_rooms[room_id] = surfaces
            else:
                print(f"[ERROR] Room {room_id} is not watertight.")
                invalid_rooms[room_id] = surfaces
        except Exception as e:
            print(f"[ERROR] Failed to build mesh for Room {room_id}: {e}")
            invalid_rooms[room_id] = surfaces

    print(f"\n✅ Valid rooms: {len(valid_rooms)}")
    print(f"❌ Invalid rooms: {len(invalid_rooms)}")

    return valid_rooms, invalid_rooms

def surface_area_3d(coords):
    if len(coords) < 3:
        return 0.0

    area = 0.0
    p0 = np.array(coords[0])

    for i in range(1, len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])
        edge1 = p1 - p0
        edge2 = p2 - p0
        cross = np.cross(edge1, edge2)
        triangle_area = np.linalg.norm(cross) / 2.0
        area += triangle_area

    return area


## Schedules
def make_schedule_fields(name, schedule_type_limits_name, lines):
    return {
        "name": name,
        "schedule_type_limits_name": schedule_type_limits_name,
        **{f"field_{i+1}": line for i, line in enumerate(lines)}
    }

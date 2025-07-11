
import os
import sys
import os.path
from pathlib import Path
import rdflib
from rdflib import Graph
from typing import List, Dict, Any

import json

import numpy as np
import laspy

import open3d as o3d

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from shapely.geometry import Polygon, MultiPolygon

sys.path.append(r"C:\Users\oscar\anaconda3\envs\cvpr\Lib\site-packages\geomapi\utils")
import geomapi.utils as ut
from geomapi.utils import geometryutils as gmu
from geomapi.nodes import PointCloudNode
import geomapi.tools as tl

from sklearn.utils import resample
from scipy.spatial import  distance, ConvexHull,  QhullError
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from scipy.spatial import ConvexHull, QhullError

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

import context_KUL
import utils_KUL as kul


## Floors and ceilings

floor_ceilings_def =  True

if floor_ceilings_def:

    def planar_xy_hull(laz, node_ids, avg_z):

        points_2d = []
        
        for n in node_ids:
            # Since n is now a dictionary, access its class_id and object_id using dictionary keys
            idx = np.where((laz['classes'] == n['class_id']) & (laz['objects'] == n['object_id']))
            
            if idx[0].size == 0:
                continue  # Skip if no points are found for the given node_id
            
            # Extract x and y values for the selected points
            x_values = laz['x'][idx]
            y_values = laz['y'][idx]
            
            # Project to 2D by combining x and y values
            projected_points = np.column_stack((x_values, y_values))
            points_2d.extend(projected_points)
        
        # Convert list of 2D points to a numpy array
        points_2d = np.array(points_2d)
        
        # Ensure there are enough points to compute a convex hull
        if points_2d.shape[0] < 3:
            print(f"Not enough points to compute convex hull for avg_z {avg_z}\nN. of points {len(points_2d)}")
            return None
        
        try:

            hull = ConvexHull(points_2d)
            hull_area = hull.volume
            
            if hull_area >= th_hull_area:  # Only return the hull if the area is >= 1 m²
                return hull
            else:
                print(f"Hull area {hull_area:.2f} is less than 1 m² for avg_z {avg_z}")
                return None
        except QhullError:
            # Handle errors in case the convex hull computation fails
            print(f"QhullError for avg_z {avg_z}")
            return None

    def pointcloudnode_to_dict(node):
        return {
            'class_id': node.class_id,
            'object_id': node.object_id,
            'subject': node.subject
        }

    def convexhull_to_dict(hull):
        """Convert a ConvexHull object to a dictionary."""
        return {
            'points': hull.points.tolist(),  # Convert points array to a list
            'vertices': hull.vertices.tolist(),  # Convert vertices array to a list
            'simplices': hull.simplices.tolist() if hasattr(hull, 'simplices') else None,  # Convert simplices to a list if available
            'volume': hull.volume  # For 2D hulls, this is the area
        }

    def load_levels(laz, graph_path):
        # Parse the graph
        graph = Graph().parse(str(graph_path))
        nodes = tl.graph_to_nodes(graph)

        # Separate nodes by type
        ceilings_nodes = [n for n in nodes if 'ceilings' in n.subject.lower() and isinstance(n, PointCloudNode)]
        floors_nodes = [n for n in nodes if 'floors' in n.subject.lower() and isinstance(n, PointCloudNode)]
        level_nodes = [n for n in nodes if 'level' in n.subject.lower() and isinstance(n, PointCloudNode)]
        print("Ceilings:", [n.subject for n in ceilings_nodes])
        print("Floors:", [n.subject for n in floors_nodes])
        print("Levels:", [n.subject for n in level_nodes])

        # Initialize lists for merged floor and ceiling data
        floors_z = []
        ceilings_z = []
        
        # Lists to store z_min and z_max values for bounding box computation
        floors_z_bbox = []  # (z_min, z_max) for each floor
        ceilings_z_bbox = []  # (z_min, z_max) for each ceiling

        # Calculate average z-values and z_min, z_max for floors
        for n in floors_nodes:
            idx = np.where((laz.classes == n.class_id) & (laz.objects == n.object_id))
            z_values = laz.z[idx]

            if len(z_values) > 0:
                avg_z = np.mean(z_values)
                z_min = np.min(z_values)
                z_max = np.max(z_values)

                if (z_max - z_min) > t_thickness_floor:
                    print(f"Skipping node {n.subject} - z range too thick: {z_max - z_min:.2f}m")
                    continue
                
                merged = False
                for i, (existing_avg_z, floor_ids) in enumerate(floors_z):
                    if abs(existing_avg_z - avg_z) <= t_floor:
                        new_avg_z = (existing_avg_z * len(floor_ids) + avg_z) / (len(floor_ids) + 1)
                        floors_z[i] = (new_avg_z, floor_ids + [pointcloudnode_to_dict(n)])
                        floors_z_bbox[i] = (min(floors_z_bbox[i][0], z_min), max(floors_z_bbox[i][1], z_max))
                        merged = True
                        break

                if not merged:
                    floors_z.append((avg_z, [pointcloudnode_to_dict(n)]))
                    floors_z_bbox.append((z_min, z_max))  # Store z_min and z_max for this floor

        # Calculate average z-values and z_min, z_max for ceilings
        for n in ceilings_nodes:
            idx = np.where((laz.classes == n.class_id) & (laz.objects == n.object_id))
            z_values = laz.z[idx]

            if len(z_values) > 0:
                avg_z = np.mean(z_values)
                z_min = np.min(z_values)
                z_max = np.max(z_values)

                if (z_max - z_min) > t_thickness_floor:
                    continue

                merged = False
                for i, (existing_avg_z, ceiling_ids) in enumerate(ceilings_z):
                    if abs(existing_avg_z - avg_z) <= t_ceiling:
                        new_avg_z = (existing_avg_z * len(ceiling_ids) + avg_z) / (len(ceiling_ids) + 1)
                        ceilings_z[i] = (new_avg_z, ceiling_ids + [pointcloudnode_to_dict(n)])
                        ceilings_z_bbox[i] = (min(ceilings_z_bbox[i][0], z_min), max(ceilings_z_bbox[i][1], z_max))
                        merged = True
                        break

                if not merged:
                    ceilings_z.append((avg_z, [pointcloudnode_to_dict(n)]))
                    ceilings_z_bbox.append((z_min, z_max))  # Store z_min and z_max for this ceiling

        print(f'Find {len(ceilings_nodes)} ceilings after normalization {len(ceilings_z)}')
        print(f'Find {len(floors_nodes)} floors after normalization {len(floors_z)}')
        print(f'Find {len(level_nodes)} levels')

        # Compute convex hulls and bounding boxes
        floor_hulls = []
        floor_hull_vertices = []
        floor_bboxes = []

        for avg_z, floor_ids in floors_z:
            # Get the convex hull
            hull = planar_xy_hull(laz, floor_ids, avg_z)
            
            if hull is None:  # hulls with area < 1m² are already discarded in planar_xy_hull
                continue
            
            floor_hulls.append(convexhull_to_dict(hull))  # Convert hull to dict
            
            # Get the vertices of the convex hull
            vertices = hull.points[hull.vertices]
            floor_hull_vertices.append(vertices.tolist())  # Convert vertices to list
            
            # Compute the 2D bounding box
            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])

            # Extend to 3D bounding box using z_min and z_max
            z_min, z_max = None, None
            for i, (z_val, ids) in enumerate(floors_z):
                if abs(z_val - avg_z) < 1e-3:  # compare avg_z safely
                    z_min, z_max = floors_z_bbox[i]
                    break

            floor_bboxes.append(np.array([
                [min_x, min_y, z_min], [max_x, min_y, z_min],
                [max_x, max_y, z_min], [min_x, max_y, z_min],
                [min_x, min_y, z_max], [max_x, min_y, z_max],
                [max_x, max_y, z_max], [min_x, max_y, z_max]
            ]))


        ceiling_hulls = []
        ceiling_hull_vertices = []
        ceiling_bboxes = []

        for avg_z, ceiling_ids in ceilings_z:
            # Get the convex hull
            hull = planar_xy_hull(laz, ceiling_ids, avg_z)
            
            if hull is None:  
                continue
            
            ceiling_hulls.append(convexhull_to_dict(hull))  # Convert hull to dict

            # Get the vertices of the convex hull
            vertices = hull.points[hull.vertices]
            ceiling_hull_vertices.append(vertices.tolist())  # Convert vertices to list
            
            # Compute the 2D bounding box
            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
            
            # Extend to 3D bounding box using z_min and z_max
            z_min, z_max = ceilings_z_bbox[ceilings_z.index((avg_z, ceiling_ids))]

            ceiling_bboxes.append(np.array([
                [min_x, min_y, z_min], [max_x, min_y, z_min],
                [max_x, max_y, z_min], [min_x, max_y, z_min],
                [min_x, min_y, z_max], [max_x, min_y, z_max],
                [max_x, max_y, z_max], [min_x, max_y, z_max]
            ]))

            
        # Compute average z-values for floors and ceilings
        floor_avg_z_values = [avg_z for avg_z, _ in floors_z]
        ceiling_avg_z_values = [avg_z for avg_z, _ in ceilings_z]

        floor_z_avg = np.mean(floor_avg_z_values) if floor_avg_z_values else None
        ceiling_z_avg = np.mean(ceiling_avg_z_values) if ceiling_avg_z_values else None


        thicknesses = []

        for floor_bbox, ceiling_bbox in zip(floors_z_bbox, ceilings_z_bbox):
            thickness = ceiling_bbox[0] - floor_bbox[1]  # z_min of ceiling - z_max of floor
            thicknesses.append(thickness)

        
        # Create and return the dictionary
        results = {
            'floors_nodes': floors_z,  # Now contains dictionaries instead of PointCloudNode objects
            'ceilings_nodes': ceilings_z,
            'floor_z_avg': floor_z_avg,
            'ceiling_z_avg': ceiling_z_avg,
            'level_nodes': [pointcloudnode_to_dict(n) for n in level_nodes],  # Convert level nodes to dictionaries
            'floors_z_bbox': floors_z_bbox,
            'ceilings_z_bbox': ceilings_z_bbox,
            'thicknesses': thicknesses,
            'floor_hulls': floor_hulls,  # Convex hulls for floors
            'floor_hull_vertices': floor_hull_vertices,  # Hull vertices for floors
            'ceiling_hulls': ceiling_hulls,  # Convex hulls for ceilings
            'ceiling_hull_vertices': ceiling_hull_vertices,  # Hull vertices for ceilings
            'floor_bboxes': floor_bboxes,  # 3D bounding boxes for floors
            'ceiling_bboxes': ceiling_bboxes  # 3D bounding boxes for ceilings
        }

        return results

    def levels_bbox(floor_bboxes, ceiling_bboxes):

        filtered_floor_bboxes = [bbox for bbox in floor_bboxes if bbox is not None]
        filtered_ceiling_bboxes = [bbox for bbox in ceiling_bboxes if bbox is not None]

        if len(filtered_floor_bboxes) == 0 and len(filtered_ceiling_bboxes) == 0:
            print("Both floor and ceiling bounding boxes are empty.")
            return np.array([])  # Return an empty array if both are empty

        if len(filtered_floor_bboxes) == 0:
            print("Floor bounding boxes are empty, only ceiling bounding boxes will be used.")
            return np.array(filtered_ceiling_bboxes)  # Return only ceiling bboxes if floors are empty

        if len(filtered_ceiling_bboxes) == 0:
            print("Ceiling bounding boxes are empty, only floor bounding boxes will be used.")
            return np.array(filtered_floor_bboxes)  # Return only floor bboxes if ceilings are empty

        floor_bboxes_array = np.array(filtered_floor_bboxes, dtype = object)
        ceiling_bboxes_array = np.array(filtered_ceiling_bboxes, dtype = object)

        levels_bboxes = np.concatenate((floor_bboxes_array, ceiling_bboxes_array), axis = 0)

        return levels_bboxes
    
    def plot_3d_bounding_boxes(results, bboxes, thicknesses=None):
    
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        floors_nodes = results['floors_nodes']
        ceilings_nodes = results['ceilings_nodes']     

        for n in floors_nodes and ceilings_nodes:

            idx = np.where((laz.classes == n.class_id) & (laz.objects == n.object_id))
            x_values = laz.x[idx]
            y_values = laz.y[idx]
            z_values = laz.z[idx]
            coords = np.vstack((x_values, y_values, z_values)).T

            ax.scatter(coords [:, 0], coords[:, 1], coords[:,2], marker = 'o', s = 0.001, alpha = 0.10)

        for bbox in bboxes:
            if bbox.shape == (8, 3):  # Ensure bbox has 8 points each with (x, y, z)
                # Define the vertices of the bounding box
                vertices = bbox
                
                faces = [
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom face
                    [vertices[7], vertices[6], vertices[2], vertices[3]],  # Top face
                    [vertices[0], vertices[1], vertices[6], vertices[7]],  # Front face
                    [vertices[2], vertices[3], vertices[4], vertices[5]],  # Back face
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
                    [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right face
                ]
                
                poly3d = Poly3DCollection(faces, alpha=0.25, linewidths=0.5, edgecolors='r')
                ax.add_collection3d(poly3d)

        # If thicknesses are provided, plot them as well
        if thicknesses is not None and len(thicknesses) == 3:
            ax.quiver(0, 0, 0, thicknesses[0], thicknesses[1], thicknesses[2], color='b', label='Thickness')
            
            # Include thickness in axis labels
            ax.set_xlabel(f'x axis (thickness = {thicknesses[0]})', fontsize=6)
            ax.set_ylabel(f'y axis (thickness = {thicknesses[1]})', fontsize=6)
            ax.set_zlabel(f'z axis (thickness = {thicknesses[2]})', fontsize=6)
        else:
            # Default labels if thicknesses are not provided
            ax.set_xlabel('x axis', fontsize=6)
            ax.set_ylabel('y axis', fontsize=6)
            ax.set_zlabel('z axis', fontsize=6)
        
        # Set tick labels' font size
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.tick_params(axis='z', labelsize=6)

        # Optionally set axis limits based on the data
        x_limits = [np.min(bboxes[:, :, 0]) - 5, np.max(bboxes[:, :, 0]) + 5]
        y_limits = [np.min(bboxes[:, :, 1]) - 5, np.max(bboxes[:, :, 1]) + 5]
        z_limits = [np.min(bboxes[:, :, 2]), np.max(bboxes[:, :, 2])]

        ax.set_box_aspect([1, 1, 0.30])
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)

        ax.grid(True)
        ax.grid(which = 'minor', linewidth = 0.25, color = 'gray')
        plt.title('3D BBOX FOR LEVELS NODES', fontsize=10)
        # plt.legend(loc='best')
        plt.show()
    
    def flatten_and_validate(vertices):
        """
        Validates and formats vertices as a 3D numpy array with shape (N, 3).
        Filters out any non-numeric entries and ensures each entry has exactly 3 values.
        """
        try:
            cleaned_vertices = []
            
            for v in vertices:
                if isinstance(v, dict):
                    try:
                        v = [float(v['x']), float(v['y']), float(v['z'])]
                    except (TypeError, ValueError, KeyError):
                        print(f"Skipping invalid dictionary entry in vertices: {v}")
                        continue

                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
                    try:
                        v = [float(coord) for coord in v]
                        cleaned_vertices.append(v)
                    except (TypeError, ValueError):
                        print(f"Skipping invalid entry in vertices: {v}")
                        continue
            
            cleaned_vertices = np.array(cleaned_vertices)

            if cleaned_vertices.ndim != 2 or cleaned_vertices.shape[1] != 3:
                print(f"❌ Invalid shape after validation: expected (N, 3), got {cleaned_vertices.shape}")
                return None

            return cleaned_vertices

        except Exception as e:
            print(f"Validation failed for vertices: {str(e)}")
            return None


    def simplify_hull(vertices, max_vertices=40):
        """
        Simplifies vertices to a convex hull with a maximum of `max_vertices`.
        """
        try:
            if vertices is None or vertices.shape[0] < 3:
                print("Not enough points for a convex hull.")
                return None
            
            hull = ConvexHull(vertices[:, :2])
            hull_points = vertices[hull.vertices]
            
            if len(hull_points) > max_vertices:
                hull_points = hull_points[::len(hull_points) // max_vertices][:max_vertices]
            
            hull_points_3d = np.column_stack((hull_points[:, 0], hull_points[:, 1], np.mean(vertices[:, 2])))
            return hull_points_3d
        except Exception as e:
            print(f"Failed to simplify vertices to a convex hull: {str(e)}")
            return None

    def export_obj_floors_ceilings(results, output_obj_file):
        """
        Exports only floor and ceiling vertex data to an OBJ file, ignoring unrelated entries.
        """
        try:
            with open(output_obj_file, 'w') as f:
                vertex_offset = 1  # OBJ indexing starts at 1
                
                # Only process entries under 'floors' and 'ceilings'
                for level_type in ['floors_nodes', 'ceilings_nodes']:

                    if level_type not in results:
                        continue  # Skip if the expected data type is missing

                    level_data = results[level_type]
                    
                    # Ensure level_data is a list to iterate through
                    if not isinstance(level_data, list):
                        print(f"Skipping {level_type}: unsupported data format")
                        continue
                    
                    for i, element in enumerate(level_data):
                        # Validate that each element has vertices
                        vertices = element.get("vertices", []) if isinstance(element, dict) else element
                        vertices = flatten_and_validate(vertices)

                        if vertices is None or vertices.size == 0:
                            print(f"Skipping {level_type}_{i + 1}: invalid or empty vertices")
                            continue

                        # Simplify hull for the element
                        simplified_vertices = simplify_hull(vertices)
                        if simplified_vertices is None:
                            print(f"Skipping {level_type}_{i + 1} after simplification.")
                            continue

                        # Write vertices to OBJ file
                        f.write(f"# {level_type.capitalize()}_{i + 1}\n")
                        for vertex in simplified_vertices:
                            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                        # Write faces (assuming quads)
                        if len(simplified_vertices) == 4:
                            f.write(f"f {vertex_offset} {vertex_offset+1} {vertex_offset+2} {vertex_offset+3}\n")
                        vertex_offset += len(simplified_vertices)

            print(f"Floors and ceilings successfully exported to {output_obj_file}")
        except Exception as e:
            print(f"Failed to export floors and ceilings to OBJ: {str(e)}")

    def export_floors_ceilings_levels_to_json(results, output_json_file):
        try:
            with open(output_json_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Data successfully written to {output_json_file}")
        except Exception as e:
            print(f"Failed to write data to JSON: {str(e)}")

walls_computations_def = True

if walls_computations_def :

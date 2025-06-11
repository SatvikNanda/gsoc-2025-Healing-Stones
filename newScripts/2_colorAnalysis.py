import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_fragment_color_and_geometry(ply_path):
    # Load the fragment
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_colors():
        raise ValueError("Point cloud has no color information.")
    if not pcd.has_normals():
        pcd.estimate_normals()

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)

    # Step 1: Compute average color brightness
    brightness = np.mean(colors, axis=1)  # mean of R, G, B per point

    # Step 2: Identify the brightest face using normal directions
    directions = {
        'X+': np.array([1, 0, 0]),
        'X-': np.array([-1, 0, 0]),
        'Y+': np.array([0, 1, 0]),
        'Y-': np.array([0, -1, 0]),
        'Z+': np.array([0, 0, 1]),
        'Z-': np.array([0, 0, -1]),
    }

    face_brightness = {}
    for key, direction in directions.items():
        dot_product = np.dot(normals, direction)
        mask = dot_product > 0.9  # nearly aligned with direction
        if np.sum(mask) > 0:
            avg_brightness = np.mean(brightness[mask])
            face_brightness[key] = avg_brightness
        else:
            face_brightness[key] = 0

    # Step 3: Identify the face with maximum brightness
    likely_inner_face = max(face_brightness, key=face_brightness.get)

    # Step 4: Classify based on overall outer color
    avg_rgb = np.mean(colors, axis=0)
    classification = "Corner" if avg_rgb[1] > avg_rgb[0] else "Middle"  # greenish = yellowish

    return {
        "average_color": avg_rgb,
        "likely_inner_face": likely_inner_face,
        "classification": classification,
        "face_brightness": face_brightness
    }

# Example usage:
fragment_dir = "data/fragments"  # replace with your local directory path
results = {}
for file in os.listdir(fragment_dir):
    if file.endswith(".ply"):
        path = os.path.join(fragment_dir, file)
        results[file] = analyze_fragment_color_and_geometry(path)

import pandas as pd


# Save the output to a CSV file instead
df.to_csv("fragment_analysis_results.csv", index=True)
print("Results saved to fragment_analysis_results.csv")



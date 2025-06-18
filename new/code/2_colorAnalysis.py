import open3d as o3d
import numpy as np
import pandas as pd
import os

def analyze_fragment_brightness_contrast(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_colors():
        raise ValueError("Point cloud has no color information.")
    if not pcd.has_normals():
        pcd.estimate_normals()

    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    brightness = np.mean(colors, axis=1)

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
        mask = dot_product > 0.9
        face_brightness[key] = np.mean(brightness[mask]) if np.any(mask) else 0

    likely_inner_face = max(face_brightness, key=face_brightness.get)
    avg_rgb = np.mean(colors, axis=0)
    brightness_range = max(face_brightness.values()) - min(face_brightness.values())

    return {
        "average_color_R": avg_rgb[0],
        "average_color_G": avg_rgb[1],
        "average_color_B": avg_rgb[2],
        "likely_inner_face": likely_inner_face,
        "brightness_range": brightness_range,
        **face_brightness
    }

# 📁 CHANGE THIS TO YOUR LOCAL FRAGMENTS DIRECTORY
fragment_dir = "data/fragments"

results = {}
for file in os.listdir(fragment_dir):
    if file.lower().endswith(".ply"):
        path = os.path.join(fragment_dir, file)
        results[file] = analyze_fragment_brightness_contrast(path)

# Create DataFrame and sort by brightness range ascending (flatter/middle-like pieces first)
df = pd.DataFrame.from_dict(results, orient='index')
df_sorted = df.sort_values(by="brightness_range", ascending=True)
df_sorted.to_csv("fragment_color_reranked.csv", index=True)
print("✅ Reranked fragments saved to fragment_color_reranked.csv")

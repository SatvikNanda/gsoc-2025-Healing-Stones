"""
A NumPy array is a fast, efficient, and structured way to store numerical data in Python.
.ply is good for human inspection or rendering.
.npy is like converting the image to a pixel matrix for machine learning.
"""

import open3d as o3d
import numpy as np
import os

input_folder = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/shrinkwrap"
output_folder = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/dataset/numpyArray"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if not file.endswith(".ply"):
        continue

    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file.replace(".ply", ".npy"))

    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    np.save(output_path, points)
    print(f"[SAVED] {file} → {output_path} ({points.shape[0]} points)")

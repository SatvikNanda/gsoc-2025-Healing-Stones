import open3d as o3d
import numpy as np
import pandas as pd
import os

def analyze_bounding_box_ratios(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()  # returns [x, y, z] dimensions

    l, w, h = extent[0], extent[1], extent[2]
    max_dim = max(extent)
    min_dim = min(extent)
    aspect_ratio = max_dim / min_dim if min_dim > 0 else 0

    return {
        "length_x": l,
        "width_y": w,
        "height_z": h,
        "aspect_ratio": aspect_ratio
    }

# 📁 CHANGE THIS to your actual fragment directory
fragment_dir = "data/fragments"

results = {}

print("🔍 Starting aspect ratio analysis...\n")

for file in os.listdir(fragment_dir):
    if file.lower().endswith(".ply"):
        path = os.path.join(fragment_dir, file)
        results[file] = analyze_bounding_box_ratios(path)
        print(f"✅ Processed: {file}")

# 📊 Convert to DataFrame and sort by aspect ratio (low to high)
df = pd.DataFrame.from_dict(results, orient='index')
df_sorted = df.sort_values(by="aspect_ratio", ascending=True)

# 💾 Save to CSV
df_sorted.to_csv("fragment_aspect_ratio_reranked.csv", index=True)
print("\n📦 Reranked fragments saved to fragment_aspect_ratio_reranked.csv")

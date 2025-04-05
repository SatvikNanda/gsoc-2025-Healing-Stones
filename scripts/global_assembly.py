import open3d as o3d
import pandas as pd
import numpy as np
import os
import copy
from icp_align2 import load_and_preprocess_pcd, run_icp

# === Config ===
DATA_PATH = "data/fragments"
CSV_PATH = "sorted_icp_results.csv"
OUTPUT_PATH = "global_assembly.ply"

# === Helper to load point cloud ===
def load_raw_pcd(filename):
    file_path = os.path.join(DATA_PATH, filename)
    return o3d.io.read_point_cloud(file_path)

# === Step 1: Read sorted ICP results ===
df = pd.read_csv(CSV_PATH)

# === Step 2: Pick base fragment ===
# Strategy: Fragment that appears most in top 20 matches
top_matches = df.head(20)
base_fragment = top_matches['Source'].mode()[0]
print(f"✅ Base fragment selected: {base_fragment}")

# === Step 3: Initialize global assembly ===
assembled_cloud = load_raw_pcd(base_fragment)

# Keep track of fragments already added
added_fragments = set([base_fragment])

# === Step 4: Start chaining fragments ===
for index, row in top_matches.iterrows():
    source = row['Source']
    target = row['Target']

    # Skip if both fragments already added
    if source in added_fragments and target in added_fragments:
        continue

    # Determine the new fragment to add
    if source in added_fragments:
        new_fragment_file = target
    elif target in added_fragments:
        new_fragment_file = source
    else:
        continue  # Skip pairs with no connection to current assembly

    print(f"\n🔗 Adding fragment: {new_fragment_file}")

    # Load and preprocess point clouds
    new_raw, new_down = load_and_preprocess_pcd(os.path.join(DATA_PATH, new_fragment_file))
    assembled_down = assembled_cloud.voxel_down_sample(voxel_size=1.0)
    assembled_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )

    # Perform ICP
    result = run_icp(new_down, assembled_down, new_raw, assembled_cloud, method="point_to_plane")

    # Apply transformation to new fragment
    transformed_new = copy.deepcopy(new_raw)
    transformed_new.transform(result.transformation)

    # Merge point clouds
    assembled_cloud += transformed_new

    # Optionally visualize assembly progress
    # o3d.visualization.draw_geometries([assembled_cloud])

    # Mark fragment as added
    added_fragments.add(new_fragment_file)

print("\n✅ Global assembly completed.")

# === Step 5: Save final assembly ===
o3d.io.write_point_cloud(OUTPUT_PATH, assembled_cloud)
print(f"✅ Global assembly saved to '{OUTPUT_PATH}'")

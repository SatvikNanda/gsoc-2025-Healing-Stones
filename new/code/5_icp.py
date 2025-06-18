import open3d as o3d
import pandas as pd
import numpy as np
import os
import copy

# === ROTATION LOGIC BASED ON LIKELY INNER FACE ===
face_to_rotation = {
    "X+": (0, 0, 0),
    "X-": (0, 0, np.pi),
    "Y+": (-np.pi / 2, 0, 0),
    "Y-": (np.pi / 2, 0, 0),
    "Z+": (0, np.pi / 2, 0),
    "Z-": (0, -np.pi / 2, 0)
}

def rotate_to_inner_face(pcd, face):
    if face not in face_to_rotation:
        return pcd
    rx, ry, rz = face_to_rotation[face]
    R = pcd.get_rotation_matrix_from_xyz((rx, ry, rz))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

# === LOADING & PREPROCESSING ===
def load_and_preprocess(file_path, face, voxel_size=1.0):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = rotate_to_inner_face(pcd, face)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )
    return pcd, pcd_down

# === ICP ALIGNMENT FUNCTION ===
def run_icp_point_to_plane(source_down, target_down, source_raw):
    threshold = 2.0
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    aligned = copy.deepcopy(source_raw)
    aligned.transform(result.transformation)
    return result.fitness, result.inlier_rmse, aligned

# === MAIN SCRIPT CONFIG ===
DATA_PATH = "data/fragments"
CSV_PATH = "final_fragment_ranking_combined.csv"
OUTPUT_CSV = "ranked_icp_with_orientation.csv"

df = pd.read_csv(CSV_PATH)
filenames = df.sort_values(by="combined_rank")["filename"].tolist()
faces = df.set_index("filename")["likely_inner_face"].to_dict()

results = []

for i in range(len(filenames) - 1):
    source_file = filenames[i + 1]
    target_file = filenames[i]

    source_face = faces[source_file]
    target_face = faces[target_file]

    print(f"Aligning {source_file} → {target_file} | using faces {source_face} → {target_face}")

    source_raw, source_down = load_and_preprocess(os.path.join(DATA_PATH, source_file), source_face)
    target_raw, target_down = load_and_preprocess(os.path.join(DATA_PATH, target_file), target_face)

    fitness, rmse, _ = run_icp_point_to_plane(source_down, target_down, source_raw)

    results.append({
        "Source": source_file,
        "Target": target_file,
        "Fitness": round(fitness, 5),
        "RMSE": round(rmse, 5),
        "SourceInnerFace": source_face,
        "TargetInnerFace": target_face
    })

# Save results
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved ICP with orientation results to {OUTPUT_CSV}")


import torch
import numpy as np
import open3d as o3d
from load_npy import FragmentPairDataset
from siamese_net import SiameseMLP

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseMLP().to(device)
model.load_state_dict(torch.load("C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/observations/trained_siamese_chamfer.pth", map_location=device))
model.eval()

# Load dataset
csv_path = r"C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/blender/pairwise_position.csv"
npy_path = r"C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/dataset/numpyArray"
dataset = FragmentPairDataset(csv_file=csv_path, npy_folder=npy_path)

# Choose a sample index to visualize
index = 0
sample = dataset[index]
src = sample['source_points'].unsqueeze(0).to(device)  # (1, 1024, 3)
tgt = sample['target_points'].cpu().numpy()           # (1024, 3)

with torch.no_grad():
    tgt_tensor = sample['target_points'].unsqueeze(0).to(device)
    T_pred = model(src, tgt_tensor).squeeze(0).cpu().numpy()

# Transform source points
src_transformed = (T_pred[:3, :3] @ sample['source_points'].T.numpy()) + T_pred[:3, 3:4]
src_transformed = src_transformed.T

# Open3D Visualization
pcd_src = o3d.geometry.PointCloud()
pcd_src.points = o3d.utility.Vector3dVector(src_transformed)
pcd_src.paint_uniform_color([1, 0, 0])  # Red

pcd_tgt = o3d.geometry.PointCloud()
pcd_tgt.points = o3d.utility.Vector3dVector(tgt)
pcd_tgt.paint_uniform_color([0, 1, 0])  # Green

o3d.visualization.draw_geometries([pcd_src, pcd_tgt])

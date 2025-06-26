"""
manual blender alignments are the ground truth. dataset uses those alignments to supervise learning.


"""


import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class FragmentPairDataset(Dataset):
    def __init__(self, csv_file, npy_folder, num_points=1024, transform=None):
        self.pairs_df = pd.read_csv(csv_file)
        self.npy_folder = npy_folder
        self.num_points = num_points
        self.transform = transform

    def __len__(self):
        return len(self.pairs_df)

    def _sample_points(self, points):
        if len(points) >= self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
        return points[indices]

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        src_name, tgt_name = row['source'], row['target']

        src_path = os.path.join(self.npy_folder, f"{src_name}.npy")
        tgt_path = os.path.join(self.npy_folder, f"{tgt_name}.npy")

        src_points = np.load(src_path)
        tgt_points = np.load(tgt_path)

        src_sampled = self._sample_points(src_points)
        tgt_sampled = self._sample_points(tgt_points)

        T_gt = row[[f"T_gt_{i}{j}" for i in range(4) for j in range(4)]].values.astype(np.float32).reshape(4, 4)

        return {
            'source_points': torch.tensor(src_sampled, dtype=torch.float32),
            'target_points': torch.tensor(tgt_sampled, dtype=torch.float32),
            'T_gt': torch.tensor(T_gt, dtype=torch.float32),
            'source_name': src_name,
            'target_name': tgt_name
        }


dataset = FragmentPairDataset(
    csv_file="C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/blender/pairwise_position.csv",
    npy_folder="C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/dataset/numpyArray"
)

sample = dataset[0]
print(sample['source_points'].shape)
print(sample['T_gt'])
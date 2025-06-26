"""
manual blender alignments are the ground truth. dataset uses those alignments to supervise learning.


"""


import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class FragmentPairDataset(Dataset):
    def __init__(self, csv_file, npy_folder, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file containing columns:
                            source, target, and T_gt_ij values (16 of them)
            npy_folder (str): Folder containing .npy point cloud files
            transform (callable, optional): Optional transformation to apply
        """
        self.pairs_df = pd.read_csv(csv_file)
        self.npy_folder = npy_folder
        self.transform = transform

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        source_name = row['source']
        target_name = row['target']

        # Load the .npy files
        source_path = os.path.join(self.npy_folder, f"{source_name}.npy")
        target_path = os.path.join(self.npy_folder, f"{target_name}.npy")

        source_points = np.load(source_path)
        target_points = np.load(target_path)

        # Get 4x4 ground truth transformation matrix
        T_gt = row[[f"T_gt_{i}{j}" for i in range(4) for j in range(4)]].values.astype(np.float32).reshape(4, 4)

        sample = {
            'source_points': source_points,
            'target_points': target_points,
            'T_gt': T_gt,
            'source_name': source_name,
            'target_name': target_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


dataset = FragmentPairDataset(
    csv_file="C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/blender/pairwise_position.csv",
    npy_folder="C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/dataset/numpyArray"
)

sample = dataset[0]
print(sample['source_points'].shape)
print(sample['T_gt'])
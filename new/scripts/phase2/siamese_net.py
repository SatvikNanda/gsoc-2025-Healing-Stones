import torch
from torch.utils.data import DataLoader
from load_npy import FragmentPairDataset

# === Define SiameseMLP ===
import torch.nn as nn

class SiameseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16)  # 4x4 transformation matrix
        )

    def forward(self, pc1, pc2):
        feat1 = self.encoder(pc1).mean(dim=1)  # (B, 256)
        feat2 = self.encoder(pc2).mean(dim=1)  # (B, 256)
        combined = torch.cat([feat1, feat2], dim=1)
        out = self.fc(combined)
        return out.view(-1, 4, 4)

# === Chamfer Loss & Transformation ===
def chamfer_distance(pc1, pc2):
    x_exp = pc1.unsqueeze(2)  # (B, N, 1, 3)
    y_exp = pc2.unsqueeze(1)  # (B, 1, M, 3)
    dist = torch.norm(x_exp - y_exp, dim=-1)  # (B, N, M)
    return torch.min(dist, dim=2)[0].mean(1) + torch.min(dist, dim=1)[0].mean(1)

def transform_point_cloud(points, T):
    B, N, _ = points.shape
    ones = torch.ones((B, N, 1), device=points.device)
    homo_points = torch.cat([points, ones], dim=2)  # (B, N, 4)
    return torch.bmm(homo_points, T.transpose(1, 2))[:, :, :3]

# === Load Data ===
dataset = FragmentPairDataset(
    csv_file="C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/blender/pairwise_position.csv",
    npy_folder="C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/dataset/numpyArray",
    num_points=1024
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# === Train Model ===
model = SiameseMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1, 51):
    total_loss = 0
    for batch in dataloader:
        src = batch['source_points']  # (B, 1024, 3)
        tgt = batch['target_points']  # (B, 1024, 3)

        optimizer.zero_grad()
        T_pred = model(src, tgt)  # (B, 4, 4)
        src_transformed = transform_point_cloud(src, T_pred)
        loss = chamfer_distance(src_transformed, tgt).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch}/50] Chamfer Loss: {total_loss / len(dataloader):.4f}")

# === Save Model ===
torch.save(model.state_dict(), "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/observations/trained_siamese_chamfer.pth")
print("✅ Model saved successfully.")

import torch
from torch.utils.data import DataLoader
from load_npy import FragmentPairDataset
from new.scripts.phase2.siamese_net import SiameseAlignmentNet
import torch.nn.functional as F
import os

# Paths
csv_path = r"C:\Users\satvi\OneDrive\Desktop\gsoc-2025-Healing-Stones\new\blender\pairwise_position.csv"
npy_path = r"C:\Users\satvi\OneDrive\Desktop\gsoc-2025-Healing-Stones\new\dataset\numpyArray"

# Hyperparams
batch_size = 4
num_epochs = 50
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Dataloader
dataset = FragmentPairDataset(csv_file=csv_path, npy_folder=npy_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = SiameseAlignmentNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    
    for batch in dataloader:
        src = batch['source_points'].to(device)  # (B, 1024, 3)
        tgt = batch['target_points'].to(device)
        T_gt = batch['T_gt'].to(device)          # (B, 4, 4)

        T_pred = model(src, tgt)                 # (B, 4, 4)
        loss = F.mse_loss(T_pred, T_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {total_loss / len(dataloader):.6f}")

torch.save(model.state_dict(), "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/observations/trained_model.pth")
print("✅ Model saved to C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/observations/trained_model.pth")

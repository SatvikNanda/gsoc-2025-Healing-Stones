import pandas as pd
import json

# Load adjacency and transformation matrix data
with open("C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/blender/adjacent.json", 'r') as f:
    adjacency = json.load(f)

transform_df = pd.read_csv("C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/blender/transformation_matrix.csv")

# Convert transformation data into a lookup dictionary
transform_lookup = {
    row['fragment']: row.drop('fragment').values.reshape(4, 4).astype(float)
    for _, row in transform_df.iterrows()
}

# Generate pairwise dataset
rows = []
for source, targets in adjacency.items():
    for target in targets:
        if target in transform_lookup:
            row = {"source": source, "target": target}
            matrix = transform_lookup[target]
            for i in range(4):
                for j in range(4):
                    row[f"T_gt_{i}{j}"] = matrix[i][j]
            rows.append(row)

# Save to CSV
pairwise_df = pd.DataFrame(rows)
output_path = "C:/Users/satvi/OneDrive/Desktop/gsoc-2025-Healing-Stones/new/blender/pairwise_position.csv"
pairwise_df.to_csv(output_path, index=False)


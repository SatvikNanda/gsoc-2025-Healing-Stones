import pandas as pd

# Load the two ranking CSVs
color_df = pd.read_csv("fragment_color_reranked.csv", index_col=0)
shape_df = pd.read_csv("fragment_aspect_ratio_reranked.csv", index_col=0)

# Reset index to merge on 'filename'
color_df = color_df.reset_index().rename(columns={"index": "filename"})
shape_df = shape_df.reset_index().rename(columns={"index": "filename"})

# Create a rank based on brightness_range (lower = flatter)
color_df["color_rank"] = color_df["brightness_range"].rank(method="min")

# Create a rank based on aspect ratio (lower = cubish)
shape_df["shape_rank"] = shape_df["aspect_ratio"].rank(method="min")

# Merge the two datasets
merged_df = pd.merge(color_df, shape_df, on="filename")

# Final combined rank
merged_df["combined_rank"] = merged_df["color_rank"] + merged_df["shape_rank"]

# Sort and save
final_df = merged_df.sort_values(by="combined_rank", ascending=True)
final_df.to_csv("final_fragment_ranking_combined.csv", index=False)

print("✅ Combined ranking saved to final_fragment_ranking_combined.csv")

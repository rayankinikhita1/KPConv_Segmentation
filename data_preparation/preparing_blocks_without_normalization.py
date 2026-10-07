import os
import json
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# -------- CONFIG --------
input_ply = "/home/nikhita_rayanki/projects/datasets/HQ_v2/original_ply/03_1_v2.ply"   # Path to your large .ply file
output_dir = "/home/nikhita_rayanki/projects/datasets/blocks_output"    # Output directory
block_size_xy = 15.0            # 15m × 15m block size
overlap = 0.2                   # 20% overlap between blocks
os.makedirs(output_dir, exist_ok=True)

# -------- READ PLY --------
plydata = PlyData.read(input_ply)
vertex = plydata['vertex'].data

fields = vertex.dtype.names
print("Detected fields:", fields)
original_dtype = vertex.dtype

# Extract coordinates
x, y, z = vertex['x'], vertex['y'], vertex['z']
min_x, max_x = np.min(x), np.max(x)
min_y, max_y = np.min(y), np.max(y)

# Calculate step size with overlap
step_xy = block_size_xy * (1 - overlap)

# Global metadata index
blocks_index = []

# -------- SPLIT INTO BLOCKS --------
block_count = 0
for ix, i in enumerate(tqdm(np.arange(min_x, max_x, step_xy), desc="Splitting along X")):
    for iy, j in enumerate(np.arange(min_y, max_y, step_xy)):
        # Select points inside the current XY block
        mask = (
            (x >= i) & (x < i + block_size_xy) &
            (y >= j) & (y < j + block_size_xy)
        )
        block_points = vertex[mask]
        if len(block_points) == 0:
            continue

        # Z bounds and height
        z_min, z_max = np.min(block_points['z']), np.max(block_points['z'])
        block_height = z_max - z_min

        # Center block (remove large coordinate bias)
        centered_block = np.copy(block_points)
        centered_block['x'] -= np.mean(centered_block['x'])
        centered_block['y'] -= np.mean(centered_block['y'])
        centered_block['z'] -= np.mean(centered_block['z'])

        # Preserve data types and fields
        new_vertex = np.empty(len(centered_block), dtype=original_dtype)
        for name in fields:
            new_vertex[name] = centered_block[name]

        # File names
        block_name = f"block_{block_count:05d}"
        ply_path = os.path.join(output_dir, f"{block_name}.ply")

        # Save block PLY
        PlyData([PlyElement.describe(new_vertex, 'vertex')], text=False).write(ply_path)

        # Compute centroid (original coordinates)
        centroid_x = float(np.mean(block_points['x']))
        centroid_y = float(np.mean(block_points['y']))
        centroid_z = float(np.mean(block_points['z']))

        # Block metadata
        metadata = {
            "block_id": block_count,
            "ply_file": os.path.basename(ply_path),
            "num_points": int(len(centered_block)),
            "grid_index": {"ix": int(ix), "iy": int(iy)},
            "centroid_original": {
                "x": centroid_x,
                "y": centroid_y,
                "z": centroid_z
            },
            "original_bbox": {
                "min_x": float(i),
                "max_x": float(i + block_size_xy),
                "min_y": float(j),
                "max_y": float(j + block_size_xy),
                "min_z": float(z_min),
                "max_z": float(z_max)
            },
            "height_m": float(block_height)
        }

        # Save JSON
        json_path = os.path.join(output_dir, f"{block_name}.json")
        with open(json_path, "w") as jf:
            json.dump(metadata, jf, indent=4)

        # Add to index
        blocks_index.append(metadata)
        block_count += 1

# -------- SAVE GLOBAL INDEX --------
index_path = os.path.join(output_dir, "blocks_metadata_index.json")
with open(index_path, "w") as jf:
    json.dump(blocks_index, jf, indent=4)

print(f"\n✅ Done! {block_count} blocks saved in '{output_dir}'")
print(f"🗂️ Metadata index written to '{index_path}'")

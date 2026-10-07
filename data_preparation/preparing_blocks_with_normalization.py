import os
import json
import numpy as np
import glob
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import time
from datetime import datetime

# -------- CONFIGURATION --------
INPUT_DATASET_DIR = "/home/nikhita_rayanki/storage/HQ_v2/original_ply"
OUTPUT_BASE_DIR = "/home/nikhita_rayanki/storage/HQ_v2/HQ_blocks_processed/"
BLOCK_SIZE_XY = 15.0            # 15m × 15m block size
OVERLAP = 0.2                   # 20% overlap between blocks
APPLY_NORMALIZATION = True      # TS40K-style normalization
NORMALIZATION_RANGE = [0, 1]    # Normalize to [0,1]

# Create output directory structure
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# -------- TS40K NORMALIZATION FUNCTION --------
def normalize_ts40k_style(points_xyz, range_vals=[0, 1]):
    """TS40K normalize() method - Min-Max scaling to [0,1]"""
    points_xyz = np.array(points_xyz)
    
    # Get min and max for each dimension
    min_vals = np.min(points_xyz, axis=0)
    max_vals = np.max(points_xyz, axis=0)
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges = np.where(ranges == 0, 1, ranges)
    
    # Scale to [0,1] then to target range
    normalized = (points_xyz - min_vals) / ranges
    normalized = normalized * (range_vals[1] - range_vals[0]) + range_vals[0]
    
    return normalized

def process_single_ply_file(input_ply_path, output_dir):
    """Process a single PLY file into blocks"""
    print(f"\n🔄 Processing: {os.path.basename(input_ply_path)}")
    
    # Create output directory for this PLY file
    ply_name = os.path.splitext(os.path.basename(input_ply_path))[0]
    ply_output_dir = os.path.join(output_dir, ply_name)
    os.makedirs(ply_output_dir, exist_ok=True)
    
    # Read PLY file
    try:
        plydata = PlyData.read(input_ply_path)
        vertex = plydata['vertex'].data
        
        # Get file info
        file_size_gb = os.path.getsize(input_ply_path) / (1024**3)
        total_points = len(vertex)
        
        print(f"   📊 File size: {file_size_gb:.2f} GB")
        print(f"   📊 Total points: {total_points:,}")
        
    except Exception as e:
        print(f"   ❌ Error reading PLY file: {e}")
        return None
    
    fields = vertex.dtype.names
    original_dtype = vertex.dtype
    print(f"   📋 Fields: {fields}")
    
    # Extract coordinates
    x, y, z = vertex['x'], vertex['y'], vertex['z']
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)
    
    print(f"   📐 Bounds: X({min_x:.1f}, {max_x:.1f}) Y({min_y:.1f}, {max_y:.1f}) Z({min_z:.1f}, {max_z:.1f})")
    
    # Calculate step size with overlap
    step_xy = BLOCK_SIZE_XY * (1 - OVERLAP)
    
    # Estimate number of blocks
    nx_blocks = int(np.ceil((max_x - min_x) / step_xy))
    ny_blocks = int(np.ceil((max_y - min_y) / step_xy))
    estimated_blocks = nx_blocks * ny_blocks
    print(f"   🔢 Estimated blocks: {estimated_blocks}")
    
    # Process blocks
    blocks_index = []
    block_count = 0
    start_time = time.time()
    
    for ix, i in enumerate(tqdm(np.arange(min_x, max_x, step_xy), desc=f"   Processing {ply_name}")):
        for iy, j in enumerate(np.arange(min_y, max_y, step_xy)):
            # Select points inside the current XY block
            mask = (
                (x >= i) & (x < i + BLOCK_SIZE_XY) &
                (y >= j) & (y < j + BLOCK_SIZE_XY)
            )
            block_points = vertex[mask]
            
            if len(block_points) == 0:
                continue
            
            # Z bounds and height
            z_min_block, z_max_block = np.min(block_points['z']), np.max(block_points['z'])
            block_height = z_max_block - z_min_block
            
            # STEP 1: Center block (remove large coordinate bias)
            centered_block = np.copy(block_points)
            centroid_x_orig = np.mean(centered_block['x'])
            centroid_y_orig = np.mean(centered_block['y'])
            centroid_z_orig = np.mean(centered_block['z'])
            
            centered_block['x'] -= centroid_x_orig
            centered_block['y'] -= centroid_y_orig
            centered_block['z'] -= centroid_z_orig
            
            # STEP 2: Apply TS40K normalization (scale to [0,1])
            if APPLY_NORMALIZATION:
                coords = np.column_stack([centered_block['x'], centered_block['y'], centered_block['z']])
                normalized_coords = normalize_ts40k_style(coords, NORMALIZATION_RANGE)
                
                centered_block['x'] = normalized_coords[:, 0]
                centered_block['y'] = normalized_coords[:, 1]
                centered_block['z'] = normalized_coords[:, 2]
            
            # Preserve data types and fields
            new_vertex = np.empty(len(centered_block), dtype=original_dtype)
            for name in fields:
                new_vertex[name] = centered_block[name]
            
            # File names
            block_name = f"{ply_name}_block_{block_count:05d}"
            ply_path = os.path.join(ply_output_dir, f"{block_name}.ply")
            
            # Save block PLY
            PlyData([PlyElement.describe(new_vertex, 'vertex')], text=False).write(ply_path)
            
            # Block metadata
            metadata = {
                "block_id": block_count,
                "source_ply": os.path.basename(input_ply_path),
                "ply_file": f"{block_name}.ply",
                "num_points": int(len(centered_block)),
                "grid_index": {"ix": int(ix), "iy": int(iy)},
                "preprocessing": {
                    "centered": True,
                    "ts40k_normalized": APPLY_NORMALIZATION,
                    "normalization_range": NORMALIZATION_RANGE
                },
                "centroid_original": {
                    "x": float(centroid_x_orig),
                    "y": float(centroid_y_orig),
                    "z": float(centroid_z_orig)
                },
                "original_bbox": {
                    "min_x": float(i),
                    "max_x": float(i + BLOCK_SIZE_XY),
                    "min_y": float(j),
                    "max_y": float(j + BLOCK_SIZE_XY),
                    "min_z": float(z_min_block),
                    "max_z": float(z_max_block)
                },
                "height_m": float(block_height)
            }
            
            # Save individual block JSON
            json_path = os.path.join(ply_output_dir, f"{block_name}.json")
            with open(json_path, "w") as jf:
                json.dump(metadata, jf, indent=2)
            
            blocks_index.append(metadata)
            block_count += 1
    
    # Save index for this PLY file
    index_path = os.path.join(ply_output_dir, f"{ply_name}_blocks_index.json")
    with open(index_path, "w") as jf:
        json.dump(blocks_index, jf, indent=2)
    
    processing_time = time.time() - start_time
    
    # Summary for this file
    summary = {
        "source_file": os.path.basename(input_ply_path),
        "file_size_gb": file_size_gb,
        "total_input_points": total_points,
        "blocks_created": block_count,
        "processing_time_seconds": processing_time,
        "average_points_per_block": int(total_points / block_count) if block_count > 0 else 0,
        "output_directory": ply_output_dir
    }
    
    print(f"   ✅ Created {block_count} blocks in {processing_time:.1f}s")
    return summary

def main():
    """Process entire HQ dataset"""
    print("🚀 Starting HQ Dataset Blocking Pipeline")
    print(f"📁 Input: {INPUT_DATASET_DIR}")
    print(f"📁 Output: {OUTPUT_BASE_DIR}")
    print(f"⚙️  Block size: {BLOCK_SIZE_XY}m × {BLOCK_SIZE_XY}m")
    print(f"⚙️  Overlap: {OVERLAP*100}%")
    print(f"⚙️  Normalization: {APPLY_NORMALIZATION} (range: {NORMALIZATION_RANGE})")
    
    # Find all PLY files
    ply_files = glob.glob(os.path.join(INPUT_DATASET_DIR, "*.ply"))
    print(f"\n📋 Found {len(ply_files)} PLY files to process:")
    
    total_size_gb = 0
    for i, ply_file in enumerate(ply_files):
        size_gb = os.path.getsize(ply_file) / (1024**3)
        total_size_gb += size_gb
        print(f"   {i+1}. {os.path.basename(ply_file)} ({size_gb:.2f} GB)")
    
    print(f"\n📊 Total dataset size: {total_size_gb:.2f} GB")
    
    # Process each PLY file
    dataset_summary = []
    total_start_time = time.time()
    
    for i, ply_file in enumerate(ply_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(ply_files)}")
        
        summary = process_single_ply_file(ply_file, OUTPUT_BASE_DIR)
        if summary:
            dataset_summary.append(summary)
    
    # Final dataset summary
    total_processing_time = time.time() - total_start_time
    total_blocks = sum(s["blocks_created"] for s in dataset_summary)
    total_input_points = sum(s["total_input_points"] for s in dataset_summary)
    
    final_summary = {
        "processing_date": datetime.now().isoformat(),
        "dataset_path": INPUT_DATASET_DIR,
        "output_path": OUTPUT_BASE_DIR,
        "configuration": {
            "block_size_xy": BLOCK_SIZE_XY,
            "overlap": OVERLAP,
            "normalization_enabled": APPLY_NORMALIZATION,
            "normalization_range": NORMALIZATION_RANGE
        },
        "statistics": {
            "input_files_processed": len(ply_files),
            "total_blocks_created": total_blocks,
            "total_input_points": total_input_points,
            "total_processing_time_seconds": total_processing_time,
            "average_points_per_block": int(total_input_points / total_blocks) if total_blocks > 0 else 0
        },
        "file_summaries": dataset_summary
    }
    
    # Save final summary
    summary_path = os.path.join(OUTPUT_BASE_DIR, "dataset_processing_summary.json")
    with open(summary_path, "w") as jf:
        json.dump(final_summary, jf, indent=2)
    
    print(f"\n{'='*60}")
    print("🎉 DATASET PROCESSING COMPLETE!")
    print(f"📊 Total files processed: {len(ply_files)}")
    print(f"📊 Total blocks created: {total_blocks}")
    print(f"📊 Total processing time: {total_processing_time/3600:.2f} hours")
    print(f"📊 Summary saved to: {summary_path}")
    print(f"📁 Output directory: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
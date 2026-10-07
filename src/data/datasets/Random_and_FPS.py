import numpy as np
import os
import time
from typing import Tuple, Optional
from plyfile import PlyData, PlyElement
import torch
from tqdm import tqdm

def read_ply_with_labels(file_path):
    """Read PLY file with coordinates, colors, and labels"""
    print(f"📖 Reading PLY file: {file_path}")
    
    try:
        ply_data = PlyData.read(file_path)
        print(f"📋 PLY elements: {[element.name for element in ply_data.elements]}")
        
        # Get the vertex element
        vertex_element = None
        for element in ply_data.elements:
            if element.name == 'vertex':
                vertex_element = element
                break
        
        if vertex_element is None:
            raise ValueError("No 'vertex' element found in PLY file")
        
        # Get vertex data
        vertex_data = vertex_element.data
        
        # Get available field names
        available_fields = vertex_data.dtype.names
        print(f"📋 Available fields in PLY: {available_fields}")
        
        # Extract coordinates
        points = np.stack([
            vertex_data['x'],
            vertex_data['y'], 
            vertex_data['z']
        ], axis=1).astype(np.float32)
        
        # Extract RGB colors
        colors = np.stack([
            vertex_data['red'],
            vertex_data['green'],
            vertex_data['blue']
        ], axis=1).astype(np.uint8)
        
        # Extract labels - try different possible label field names
        label_fields = ['scalar_Classification', 'class', 'label', 'classification']
        labels = None
        
        for field in label_fields:
            if field in available_fields:
                labels = np.array(vertex_data[field]).astype(np.int32)
                print(f"✅ Found labels in field: '{field}'")
                break
        
        if labels is None:
            raise ValueError(f"No label field found. Available fields: {available_fields}")
        
        print(f"📊 Loaded: {points.shape[0]:,} points with {len(np.unique(labels))} unique labels")
        return points, colors, labels
        
    except Exception as e:
        print(f"❌ Error reading PLY file: {str(e)}")
        print(f"File path: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        raise

def write_ply_with_labels(file_path, points, colors, labels, stage_name="processed"):
    """Write PLY file with coordinates, colors, and labels"""
    print(f"💾 Saving {stage_name} PLY: {file_path}")
    
    # Prepare vertex data
    vertex_data = np.zeros(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('scalar_Classification', 'i4')
    ])
    
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1] 
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    vertex_data['scalar_Classification'] = labels
    
    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY file
    PlyData([vertex_element]).write(file_path)
    print(f"✅ Saved: {points.shape[0]:,} points")

def random_prefilter(points, colors, labels, target_size, seed=42):
    """Apply random pre-filtering to reduce point cloud size"""
    print(f"\n🎲 STAGE 1: Random Pre-filtering")
    print(f"Target size: {target_size:,} points")
    
    total_points = points.shape[0]
    if total_points <= target_size:
        print(f"⚠️  Point cloud already smaller than target ({total_points:,} ≤ {target_size:,})")
        return points, colors, labels
    
    # Set seed for reproducible results
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Random sampling without replacement
    indices = np.random.choice(total_points, target_size, replace=False)
    
    filtered_points = points[indices]
    filtered_colors = colors[indices]  
    filtered_labels = labels[indices]
    
    elapsed_time = time.time() - start_time
    
    print(f"⚡ Pre-filtering completed: {total_points:,} → {target_size:,} points")
    print(f"⏱️  Time: {elapsed_time:.2f} seconds")
    
    # Show label distribution after pre-filtering
    unique_labels, counts = np.unique(filtered_labels, return_counts=True)
    print(f"📊 Label distribution after pre-filtering:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(filtered_labels)) * 100
        print(f"   Class {label}: {count:,} points ({percentage:.1f}%)")
    
    return filtered_points, filtered_colors, filtered_labels

def gpu_farthest_point_sampling(points, target_samples, device='cuda'):
    """GPU-accelerated Farthest Point Sampling"""
    print(f"\n🎯 STAGE 2: GPU FPS Sampling")
    print(f"Target samples: {target_samples:,}")
    
    n_points = points.shape[0]
    if target_samples >= n_points:
        print(f"⚠️  Requested samples ({target_samples:,}) ≥ available points ({n_points:,})")
        return np.arange(n_points)
    
    start_time = time.time()
    
    # Convert to torch tensor on GPU
    points_tensor = torch.from_numpy(points).float().to(device)
    
    # Initialize
    selected_indices = torch.zeros(target_samples, dtype=torch.long, device=device)
    distances = torch.full((n_points,), float('inf'), device=device)
    
    # First point (random)
    first_idx = torch.randint(0, n_points, (1,), device=device)
    selected_indices[0] = first_idx
    
    # Update distances to first point
    current_point = points_tensor[first_idx]
    current_distances = torch.norm(points_tensor - current_point, dim=1)
    distances = torch.minimum(distances, current_distances)
    
    print("🚀 Running GPU FPS...")
    
    # Iteratively select farthest points
    for i in tqdm(range(1, target_samples), desc="FPS Progress"):
        # Select point with maximum distance
        farthest_idx = torch.argmax(distances)
        selected_indices[i] = farthest_idx
        
        # Update distances
        current_point = points_tensor[farthest_idx]
        point_diff = points_tensor - current_point
        current_distances = torch.norm(point_diff, dim=1)
        distances = torch.minimum(distances, current_distances)
    
    elapsed_time = time.time() - start_time
    
    print(f"✅ FPS completed: {n_points:,} → {target_samples:,} points")
    print(f"⏱️  Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    
    return selected_indices.cpu().numpy()

def apply_fps_with_prefiltering(points, colors, labels, 
                               prefilter_size=500000, 
                               fps_target=80000,
                               device='cuda'):
    """Complete pipeline: Pre-filtering + FPS"""
    print(f"\n🔄 COMPLETE PIPELINE: Pre-filtering + FPS")
    print(f"Input: {points.shape[0]:,} points")
    print(f"Pre-filter target: {prefilter_size:,}")
    print(f"Final FPS target: {fps_target:,}")
    
    total_start_time = time.time()
    
    # Stage 1: Random pre-filtering (if needed)
    if points.shape[0] > prefilter_size:
        pre_points, pre_colors, pre_labels = random_prefilter(
            points, colors, labels, prefilter_size
        )
    else:
        print("⚠️  Skipping pre-filtering (cloud already small enough)")
        pre_points, pre_colors, pre_labels = points, colors, labels
    
    # Stage 2: FPS sampling
    fps_indices = gpu_farthest_point_sampling(pre_points, fps_target, device)
    
    final_points = pre_points[fps_indices]
    final_colors = pre_colors[fps_indices]
    final_labels = pre_labels[fps_indices]
    
    total_elapsed = time.time() - total_start_time
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"📊 Final result: {points.shape[0]:,} → {final_points.shape[0]:,} points")
    print(f"⏱️  Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
    
    # Final label distribution
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    print(f"\n📈 Final label distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(final_labels)) * 100
        print(f"   Class {label}: {count:,} points ({percentage:.1f}%)")
    
    return final_points, final_colors, final_labels

def main():
    """Main function to test pre-filtering + FPS pipeline"""
    
    # Configuration
    INPUT_PLY = "/home/nikhita_rayanki/storage/HQ_v2/original_ply/03_1_v2.ply"
    OUTPUT_DIR = "/home/nikhita_rayanki/storage/HQ_v2/fps_testing"
    
    # Sampling parameters
    PREFILTER_SIZE = 800000  # 800K points after random pre-filtering
    FPS_TARGET = 130000       # 130K points after FPS

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 PRE-FILTERING + FPS TESTING PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Load original PLY
        points, colors, labels = read_ply_with_labels(INPUT_PLY)
        
        # Step 2: Apply complete pipeline
        final_points, final_colors, final_labels = apply_fps_with_prefiltering(
            points, colors, labels,
            prefilter_size=PREFILTER_SIZE,
            fps_target=FPS_TARGET,
            device='cuda'
        )
        
        # Step 3: Save intermediate results for comparison
        base_name = os.path.splitext(os.path.basename(INPUT_PLY))[0]
        
        # Save pre-filtered result (for comparison)
        if points.shape[0] > PREFILTER_SIZE:
            pre_points, pre_colors, pre_labels = random_prefilter(
                points, colors, labels, PREFILTER_SIZE, seed=42
            )
            prefilter_file = os.path.join(OUTPUT_DIR, f"{base_name}_prefiltered_{PREFILTER_SIZE//1000}k.ply")
            write_ply_with_labels(prefilter_file, pre_points, pre_colors, pre_labels, "pre-filtered")
        
        # Save final FPS result
        final_file = os.path.join(OUTPUT_DIR, f"{base_name}_fps_{FPS_TARGET//1000}k.ply")
        write_ply_with_labels(final_file, final_points, final_colors, final_labels, "final FPS")
        
        print(f"\n✅ SUCCESS! Files saved to: {OUTPUT_DIR}")
        print(f"📁 Pre-filtered: {base_name}_prefiltered_{PREFILTER_SIZE//1000}k.ply")
        print(f"📁 Final FPS: {base_name}_fps_{FPS_TARGET//1000}k.ply")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
import os
import glob
import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement

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
    
    return normalized, min_vals, max_vals, ranges

def remove_nan_normalize_and_save(ply_file, output_dir, apply_normalization=False):
    """
    Remove NaN values and optionally apply TS40K normalization
    """
    print(f"📖 Processing: {os.path.basename(ply_file)}")
    
    try:
        # Read PLY file
        plydata = PlyData.read(ply_file)
        vertex = plydata['vertex'].data
        df = pd.DataFrame(vertex)
        
        # Check original size
        original_count = len(df)
        print(f"   📊 Original points: {original_count:,}")
        
        # Remove rows with NaN in scalar_Classification
        if "scalar_Classification" in df.columns:
            df_clean = df.dropna(subset=["scalar_Classification"])
            clean_count = len(df_clean)
            
            if clean_count == 0:
                print(f"   ❌ All rows removed due to NaN, skipping file")
                return False
            
            nan_removed = original_count - clean_count
            if nan_removed > 0:
                print(f"   🧹 Removed {nan_removed:,} NaN rows")
            
            # Convert back to structured array
            clean_vertex = df_clean.to_records(index=False)
            clean_vertex = clean_vertex.astype(vertex.dtype)
            
            # Apply normalization if requested
            if apply_normalization:
                # Extract coordinates
                coords = np.column_stack([clean_vertex['x'], clean_vertex['y'], clean_vertex['z']])
                
                # Store original bounds for metadata
                original_bounds = {
                    'min_x': float(np.min(coords[:, 0])),
                    'max_x': float(np.max(coords[:, 0])),
                    'min_y': float(np.min(coords[:, 1])),
                    'max_y': float(np.max(coords[:, 1])),
                    'min_z': float(np.min(coords[:, 2])),
                    'max_z': float(np.max(coords[:, 2]))
                }
                
                print(f"   📐 Original bounds: X({original_bounds['min_x']:.1f}, {original_bounds['max_x']:.1f}) "
                      f"Y({original_bounds['min_y']:.1f}, {original_bounds['max_y']:.1f}) "
                      f"Z({original_bounds['min_z']:.1f}, {original_bounds['max_z']:.1f})")
                
                # Apply TS40K normalization
                normalized_coords, min_vals, max_vals, ranges = normalize_ts40k_style(coords, [0, 1])
                
                # Update coordinates in the structured array
                clean_vertex['x'] = normalized_coords[:, 0]
                clean_vertex['y'] = normalized_coords[:, 1] 
                clean_vertex['z'] = normalized_coords[:, 2]
                
                print(f"   📏 Normalized to [0,1] range")
                
                # Save normalization metadata
                metadata = {
                    'original_bounds': original_bounds,
                    'normalization_params': {
                        'min_vals': min_vals.tolist(),
                        'max_vals': max_vals.tolist(),
                        'ranges': ranges.tolist()
                    },
                    'normalization_range': [0, 1],
                    'points_processed': clean_count,
                    'nan_removed': nan_removed
                }
                
                # Save metadata as JSON
                import json
                base_name = os.path.splitext(os.path.basename(ply_file))[0]
                metadata_path = os.path.join(output_dir, f"{base_name}_normalization_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Save cleaned (and optionally normalized) PLY file
            new_ply_element = PlyElement.describe(clean_vertex, 'vertex')
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(ply_file))[0]
            if apply_normalization:
                out_filename = f"{base_name}_cleaned_normalized.ply"
            else:
                out_filename = f"{base_name}_cleaned.ply"
            
            out_path = os.path.join(output_dir, out_filename)
            PlyData([new_ply_element], text=plydata.text).write(out_path)
            
            print(f"   ✅ Saved: {out_filename}")
            print(f"   📊 Final points: {clean_count:,}")
            
            return True
            
        else:
            print(f"   ⚠️  Missing 'scalar_Classification' column, skipping")
            return False
            
    except Exception as e:
        print(f"   ❌ Error processing {ply_file}: {str(e)}")
        return False

def process_folder(input_folder, output_folder, apply_normalization=False):
    """
    Process all PLY files in a folder
    """
    print(f"🚀 Starting batch processing")
    print(f"📁 Input: {input_folder}")
    print(f"📁 Output: {output_folder}")
    print(f"📏 Apply normalization: {apply_normalization}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PLY files
    ply_files = glob.glob(os.path.join(input_folder, "*.ply"))
    print(f"\n📋 Found {len(ply_files)} PLY files")
    
    if len(ply_files) == 0:
        print("❌ No PLY files found!")
        return
    
    # Process statistics
    stats = {
        'total_files': len(ply_files),
        'processed_successfully': 0,
        'failed': 0,
        'total_original_points': 0,
        'total_final_points': 0
    }
    
    # Process each file
    for i, ply_path in enumerate(ply_files, 1):
        print(f"\n📂 Processing file {i}/{len(ply_files)}")
        success = remove_nan_normalize_and_save(ply_path, output_folder, apply_normalization)
        
        if success:
            stats['processed_successfully'] += 1
        else:
            stats['failed'] += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {stats['processed_successfully']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📁 Output directory: {output_folder}")
    
    if apply_normalization:
        print(f"📏 All coordinates normalized to [0,1] range")
        print(f"📋 Normalization metadata saved as JSON files")

if __name__ == "__main__":
    # Configuration
    input_folder = "/home/nikhita_rayanki/storage/"
    output_folder = "/home/nikhita_rayanki/storage/HQ_v2_cleaned_normalized/original_ply"
    
    # Set to True to apply TS40K normalization, False to only remove NaN
    APPLY_NORMALIZATION = False
    
    print("🏭 PLY File Cleaning and Normalization Tool")
    print("=" * 50)
    
    # Process all files
    process_folder(input_folder, output_folder, APPLY_NORMALIZATION)
#!/usr/bin/env python3
"""
Simple and clean class weights calculator for three methods: Balanced, Inverse, Critical.
Outputs a human-readable YAML file with clean formatting.
"""

import os
import numpy as np
import yaml
from collections import Counter
import argparse
from pathlib import Path

try:
    from plyfile import PlyData
except ImportError:
    print("Installing plyfile...")
    os.system("pip install plyfile")
    from plyfile import PlyData

def load_ply_file(file_path):
    """Load PLY file and extract class labels."""
    try:
        plydata = PlyData.read(file_path)
        vertex_data = plydata['vertex']
        
        # Get the actual data array
        vertex_array = vertex_data.data
        
        # Get field names from the dtype
        field_names = vertex_array.dtype.names
        
        # Try different possible label field names
        label_field = None
        for field in ['scalar_Classification', 'class', 'label', 'scalar_class', 'classification']:
            if field in field_names:
                label_field = field
                break
        
        if label_field is None:
            print(f"Warning: No label field found in {file_path}")
            print(f"Available fields: {field_names}")
            return np.array([])
        
        labels = vertex_array[label_field]
        return np.array(labels)
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.array([])

def analyze_dataset_frequencies(data_dir, file_pattern="*.ply"):
    """Analyze class frequencies across all PLY files in the dataset."""
    print(f"Analyzing dataset in: {data_dir}")
    
    # Find all PLY files
    data_path = Path(data_dir)
    ply_files = list(data_path.glob(file_pattern))
    
    if not ply_files:
        # Try in original_ply subdirectory
        original_ply_path = data_path / "original_ply"
        if original_ply_path.exists():
            ply_files = list(original_ply_path.glob(file_pattern))
            print(f"Found PLY files in original_ply subdirectory: {len(ply_files)}")
        else:
            print(f"No PLY files found matching pattern {file_pattern}")
            return None
    
    print(f"Found {len(ply_files)} PLY files")
    
    # Collect all labels
    all_labels = []
    
    for ply_file in ply_files:
        print(f"Processing: {ply_file.name}")
        labels = load_ply_file(ply_file)
        
        if len(labels) > 0:
            all_labels.extend(labels)
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"  - Found {len(labels)} points with {len(unique_labels)} unique classes")
        else:
            print(f"  - No valid labels found")
    
    if not all_labels:
        print("No labels found in any files!")
        return None
    
    # Calculate overall frequencies and convert to clean integers
    class_counts = Counter(all_labels)
    # Convert numpy types to clean Python integers
    clean_class_counts = {}
    for class_id, count in class_counts.items():
        # Convert float class IDs to integers if they are whole numbers
        if isinstance(class_id, (np.floating, float)) and float(class_id).is_integer():
            clean_class_id = int(float(class_id))
        else:
            clean_class_id = int(class_id) if isinstance(class_id, (np.integer, np.floating)) else class_id
        clean_class_counts[clean_class_id] = int(count)
    
    print(f"\nOverall dataset statistics:")
    print(f"Total points: {len(all_labels)}")
    print(f"Number of classes: {len(clean_class_counts)}")
    
    return clean_class_counts

def calculate_weights(class_counts):
    """Calculate all three types of weights and return clean Python values."""
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    counts_array = np.array(counts)
    
    n_samples = sum(counts)
    n_classes = len(classes)
    
    # 1. Balanced weights: n_samples / (n_classes * class_count)
    balanced_weights = {}
    for i, class_id in enumerate(classes):
        weight = n_samples / (n_classes * counts[i])
        balanced_weights[class_id] = float(weight)
    
    # 2. Inverse frequency weights: 1 / class_frequency
    inverse_weights = {}
    for i, class_id in enumerate(classes):
        frequency = counts[i] / n_samples
        weight = 1.0 / frequency
        inverse_weights[class_id] = float(weight)
    
    # 3. Critical weights: inverse frequency with boost for underrepresented classes
    critical_threshold = 0.05
    critical_weights = {}
    for i, class_id in enumerate(classes):
        frequency = counts[i] / n_samples
        base_weight = 1.0 / frequency
        
        if frequency < critical_threshold:
            # Apply exponential boost
            boost_factor = np.exp(2 * (critical_threshold - frequency) / critical_threshold)
            weight = base_weight * boost_factor
        else:
            weight = base_weight
        
        critical_weights[class_id] = float(weight)
    
    return balanced_weights, inverse_weights, critical_weights

def normalize_weights(weights_dict):
    """Normalize weights so minimum weight is 1.0."""
    min_weight = min(weights_dict.values())
    normalized = {}
    for class_id, weight in weights_dict.items():
        normalized[class_id] = round(weight / min_weight, 4)
    return normalized

def create_clean_yaml_output(class_counts, balanced_weights, inverse_weights, critical_weights):
    """Create a clean, readable YAML structure."""
    
    # Normalize all weights
    balanced_norm = normalize_weights(balanced_weights)
    inverse_norm = normalize_weights(inverse_weights)
    critical_norm = normalize_weights(critical_weights)
    
    # Create clean output structure
    output = {
        'dataset_summary': {
            'total_points': sum(class_counts.values()),
            'num_classes': len(class_counts),
            'class_distribution': dict(sorted(class_counts.items()))
        },
        'class_weights': {
            'balanced': {
                'description': 'Balanced weights: n_samples / (n_classes * class_count)',
                'weights': dict(sorted(balanced_norm.items()))
            },
            'inverse_frequency': {
                'description': 'Inverse frequency weights: 1 / class_frequency',
                'weights': dict(sorted(inverse_norm.items()))
            },
            'critical': {
                'description': 'Critical weights with 5% threshold for exponential boost',
                'weights': dict(sorted(critical_norm.items()))
            }
        },
        'usage_examples': {
            'pytorch_loss': 'Use weights with torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))',
            'config_format': 'Copy weights list to your model config file'
        }
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description="Calculate clean class weights for three methods")
    parser.add_argument("--data_dir", type=str, 
                       default="/home/nikhita_rayanki/storage/HQ_v2_cleaned",
                       help="Directory containing PLY files")
    parser.add_argument("--output", type=str,
                       default="clean_class_weights.yaml",
                       help="Output YAML file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLEAN THREE-METHOD CLASS WEIGHT CALCULATOR")
    print("=" * 60)
    
    # Analyze dataset
    class_counts = analyze_dataset_frequencies(args.data_dir)
    
    if class_counts is None:
        print("Failed to analyze dataset. Exiting.")
        return
    
    # Display class distribution
    print("\nClass Distribution:")
    print("-" * 50)
    total_points = sum(class_counts.values())
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_points) * 100
        print(f"Class {class_id:2d}: {count:10,} points ({percentage:6.2f}%)")
    
    # Calculate weights
    print("\nCalculating weights...")
    balanced_weights, inverse_weights, critical_weights = calculate_weights(class_counts)
    
    # Normalize weights
    balanced_norm = normalize_weights(balanced_weights)
    inverse_norm = normalize_weights(inverse_weights)
    critical_norm = normalize_weights(critical_weights)
    
    # Display weight comparison
    print("\nNormalized Weight Comparison:")
    print("-" * 70)
    print(f"{'Class':<6} {'Balanced':<12} {'Inverse':<12} {'Critical':<12}")
    print("-" * 70)
    
    for class_id in sorted(class_counts.keys()):
        balanced_w = balanced_norm[class_id]
        inverse_w = inverse_norm[class_id]
        critical_w = critical_norm[class_id]
        print(f"{class_id:<6} {balanced_w:<12.4f} {inverse_w:<12.4f} {critical_w:<12.4f}")
    
    # Create and save clean YAML
    output_data = create_clean_yaml_output(class_counts, balanced_weights, inverse_weights, critical_weights)
    
    with open(args.output, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"\nResults saved to: {args.output}")
    print("\nWeight arrays ready for use:")
    print("-" * 40)
    
    # Print weights in array format for easy copying
    balanced_array = [balanced_norm[i] for i in sorted(balanced_norm.keys())]
    inverse_array = [inverse_norm[i] for i in sorted(inverse_norm.keys())]
    critical_array = [critical_norm[i] for i in sorted(critical_norm.keys())]
    
    print("Balanced weights array:")
    print(f"  {balanced_array}")
    print("\nInverse frequency weights array:")
    print(f"  {inverse_array}")
    print("\nCritical weights array:")
    print(f"  {critical_array}")

if __name__ == "__main__":
    main()
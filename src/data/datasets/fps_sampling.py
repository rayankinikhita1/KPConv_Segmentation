
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, Tuple

def gpu_farthest_point_sampling(points, target_samples, device='cuda'):
    """
    Your working GPU-accelerated Farthest Point Sampling implementation
    """
    if isinstance(points, np.ndarray):
        point_tensor = torch.from_numpy(points).float().to(device)
    else:
        point_tensor = points.float().to(device)
    
    total_points = point_tensor.shape[0]
    if target_samples >= total_points:
        return torch.arange(total_points, device=device)

    selected_indices = torch.zeros(target_samples, dtype=torch.long, device=device)
    min_distances = torch.full((total_points,), float('inf'), device=device)
    
    # Random starting point
    initial_idx = torch.randint(0, total_points, (1,), device=device).item()
    selected_indices[0] = initial_idx
    
    # Initialize distances from first point
    initial_point = point_tensor[initial_idx:initial_idx+1]
    min_distances = torch.norm(point_tensor - initial_point, dim=1)
    
    processing_batch = min(50000, total_points)
    
    for iteration in tqdm(range(1, target_samples), desc="GPU FPS Progress", leave=False):
        # Select farthest point
        furthest_idx = torch.argmax(min_distances).item()
        selected_indices[iteration] = furthest_idx
        furthest_point = point_tensor[furthest_idx:furthest_idx+1]
        
        # Update distances in batches
        for batch_start in range(0, total_points, processing_batch):
            batch_end = min(batch_start + processing_batch, total_points)
            batch_points = point_tensor[batch_start:batch_end]
            
            # Compute distances from new point
            point_diff = batch_points - furthest_point
            current_distances = torch.norm(point_diff, dim=1)
            
            # Update with minimum distance
            min_distances[batch_start:batch_end] = torch.minimum(
                min_distances[batch_start:batch_end], current_distances)
    
    return selected_indices.cpu().numpy()


def apply_fps_subsampling(points: np.ndarray, 
                         features: Optional[np.ndarray] = None, 
                         labels: Optional[np.ndarray] = None, 
                         sample_count: int = 120000) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply GPU FPS subsampling - KPConv pipeline compatible
    
    Args:
        points: Point coordinates (N, 3)
        features: Point features like colors (N, F) 
        labels: Point labels (N,)
        sample_count: Target number of samples from config
    
    Returns:
        Tuple of (subsampled_points, subsampled_features, subsampled_labels)
    """
    total_points = points.shape[0]
    target_samples = min(sample_count, total_points)
    
    if target_samples >= total_points:
        return points, features, labels
    
    print(f"🎯 GPU FPS: {total_points:,} → {target_samples:,} points")
    
    # Apply GPU FPS sampling
    selected_indices = gpu_farthest_point_sampling(points, target_samples, 'cuda')
    
    # Extract subsampled data
    subsampled_points = points[selected_indices]
    subsampled_features = features[selected_indices] if features is not None else None
    subsampled_labels = labels[selected_indices] if labels is not None else None
    
    print(f"✅ FPS completed: {subsampled_points.shape[0]:,} points selected")
    
    return subsampled_points, subsampled_features, subsampled_labels
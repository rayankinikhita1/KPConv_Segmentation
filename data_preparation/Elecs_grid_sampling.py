
import os
import sys
import numpy as np
from os.path import join, exists, dirname, abspath

# Add the project root directory to Python path
notebook_dir = dirname(abspath('__file__'))
project_root = dirname(dirname(notebook_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from src.data.utils.ply import read_ply, write_ply
from src.data.datasets.common import grid_subsampling


def subsample_ply_folder(input_folder, output_folder, sampleDl=0.1):
    """
    Subsample all .ply files in a folder using grid_subsampling.

    Parameters:
        input_folder (str): Path to folder containing original .ply files
        output_folder (str): Path where subsampled files will be saved
        sampleDl (float): Grid subsampling resolution
    """
    if not exists(output_folder):
        os.makedirs(output_folder)

    ply_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]
    print(f"Found {len(ply_files)} PLY files to subsample")

    for f in ply_files:
        input_path = join(input_folder, f)
        output_path = join(output_folder, f)

        print(f"\nProcessing {f} ...")

        # Load cloud
        data = read_ply(input_path)
        points = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)

        colors = None
        labels = None

        # Use dtype.names for structured array checks
        if data.dtype.names is not None:
            if all(field in data.dtype.names for field in ['red', 'green', 'blue']):
                colors = np.vstack((data['red'], data['green'], data['blue'])).T.astype(np.float32)
            if 'class' in data.dtype.names:
                labels = np.array(data['class'], dtype=np.int32)
            elif 'scalar_Classification' in data.dtype.names:
                labels = np.array(data['scalar_Classification'], dtype=np.int32)
            elif 'label' in data.dtype.names:  # alternative naming
                labels = np.array(data['label'], dtype=np.int32)

        # Subsample
        sub_points, sub_colors, sub_labels = grid_subsampling(
            points, features=colors, labels=labels, sampleDl=sampleDl
        )

        # Enforce correct dtypes
        sub_points = np.asarray(sub_points, dtype=np.float32)
        if sub_colors is not None:
            sub_colors = np.asarray(sub_colors, dtype=np.float32) / 255.0
        if sub_labels is not None:
            sub_labels = np.squeeze(np.asarray(sub_labels, dtype=np.int32))

        # Save ply
        write_ply(
            output_path,
            [sub_points, sub_colors, sub_labels],
            ['x', 'y', 'z', 'red', 'green', 'blue', 'class']
        )

        print(f"Saved subsampled cloud to {output_path} ({sub_points.shape[0]} points)")


if __name__ == "__main__":
    input_folder = "/home/nikhita_rayanki/projects/datasets/Elecs"
    output_folder = "/home/nikhita_rayanki/projects/datasets/grid_subsampled_files/Elecs_0.02"
    subsample_ply_folder(input_folder, output_folder, sampleDl=0.02)

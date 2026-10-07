
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from src.data.datasets.S3DIS_Elecs import *
from torch.utils.data import DataLoader
from src.data.utils.tester import ModelTester as CustomModelTester
from src.models.architectures import KPFCNN
import time

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

   

    # Choose to test on validation or test split
    on_val = False

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

   
    from omegaconf import OmegaConf

    # Load the exact config used during training from the checkpoint logs
    checkpoint_file = '/home/nikhita_rayanki/projects/pointcloud-processing-pipeline/KPConv-lightning-hydra/logs/train/runs/2025-10-16_13-17-54/checkpoints/last.ckpt'
    
    # Extract checkpoint directory automatically from checkpoint file path
    checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_file))  # Remove /checkpoints/last.ckpt
    config_path = os.path.join(checkpoint_dir, '.hydra', 'config.yaml')
    
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Loading config from: {config_path}")
    
    loaded_config = OmegaConf.load(config_path)
    
    # The config from checkpoint has everything under 'common'
    # We need to create a config object that the dataset expects
    from omegaconf import DictConfig
    
    # Create a config object with the expected structure using the common section
    config = DictConfig(loaded_config.common)
    
    # Modify config to save to storage disk with absolute path to avoid OS disk space issues
    config.saving_path = '/home/nikhita_rayanki/storage/test_results/elecs_test_results'

    # Update paths if needed
    config.data_dir = "/home/nikhita_rayanki/storage/HQ_v2_cleaned"
    config.expected_N = 12000
    
    # Enable saving of test results
    config.saving = True
    
    print(f"Using config with name: {config.name}")
    print(f"Number of classes: {len(config.label_to_names)}")
    print(f"Saving enabled: {config.saving}")
    
    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 10

    #testing on trained data so change [0, 0, 0, 2, 0, 1, 1, 0, 0] to [2, 2, 2, 0, 2, 1, 1, 2, 2] 
    #config.all_splits = [2, 2, 2, 0, 2, 1, 1, 2, 2]  # Your desired splits

    # Create the directory if it doesn't exist
    os.makedirs(config.saving_path, exist_ok=True)
    
    print(f"Set saving_path to: {config.saving_path}")
    print(f"Saving directory exists: {os.path.exists(config.saving_path)}")
    print(f"Current working directory: {os.getcwd()}")
    

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'
    num_votes=10

    test_dataset = S3DISDataset(config, set=set, use_potentials=True)
    test_sampler = S3DISSampler(test_dataset)
    collate_fn = S3DISCollate
    

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
   
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    

    # Define a visualizer class
    chosen_chkp = checkpoint_file
    print(f"Loading checkpoint from: {chosen_chkp}")
    tester = CustomModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # # Training
    
    tester.cloud_segmentation_test(net, test_loader, config)
    
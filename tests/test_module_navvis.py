
import signal
import os
import numpy as np
import sys
import torch
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Dataset
from src.data.datasets.S3DIS_navvis import *
from torch.utils.data import DataLoader
from src.data.utils.tester import ModelTester
from src.models.architectures import KPFCNN

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

    #TODO: Get the path from path/default.yaml
    defaults = {"paths": {"s3dis_navvis_data_dir": "C:/Projects/Hydro-Qubec/DataSets/ClientDataset/data"}}

    #TODO: Relative path to the config file
    config_path = os.path.join(project_root, "configs", "common", "navvis.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(defaults, config)
    
    #TODO: # Correct way to update a value
    config.expected_N = 5000    

    #TODO: # Relative Path to the chosen checkpoint, should select the last saved checkpoint
    # Need logic to check if the path exists and the checkpoint is navvis dataset trained
    chosen_chkp = os.path.join(project_root, 'logs/train/runs/2025-09-22_17-57-39/checkpoints/last.ckpt')

    print (f"------------{config=}")
    
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

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        dataset_split = 'validation'
    else:
        dataset_split = 'test'
    num_votes=10

    test_dataset = S3DISDataset(config, set=dataset_split, use_potentials=True)
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
    
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # # Training
    
    tester.cloud_segmentation_test(net, test_loader, config)
    
#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/train_s3dis_elecs.py common.in_radius=10.0 common.first_subsampling_dl=0.3 common.expected_N=20000

# Best so far
# python src/train_s3dis_elecs.py common.in_radius=15.0 common.first_subsampling_dl=0.3

# Best Contender
# python src/train_s3dis_elecs.py common.in_radius=15.0 common.first_subsampling_dl=0.3 common.expected_N=15000

python src/train_s3dis_elecs.py common.in_radius=17.0 common.first_subsampling_dl=0.3 common.expected_N=15000


# python src/train_s3dis_elecs.py common.in_radius=10.0 common.first_subsampling_dl=0.6




#!/bin/bash

# module add openmind/miniconda/4.0.5-python3
module add openmind/miniconda/2020-01-29-py3.7
# module add nklab/pytorch/1.1.0_cuda10.0.130

module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files

source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch

ipython timing_tests.py



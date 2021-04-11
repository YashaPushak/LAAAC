#!/bin/bash

# This will vary depending on your environment
export PATH="/global/scratch/ypushak/anaconda3/envs/ray-env/bin:$PATH"
# Optionally direct where ray creates files it uses internally
HOME="/global/scratch/ypushak/ray"

# While I typically run everything through a slurm batch job, you can also just
# run things locally as well.

# You can also specify scenario file arguments as command line arguments here.
run_directory=$(pwd)
optimizer='CQA'
cpus_per_trial=1
scenario='scenario.txt'


cd ../../
echo $(pwd)

echo "python run.py --run_directory $run_directory --scenario $scenario --optimizer $optimizer"
python run.py --run_directory $run_directory --scenario $scenario --optimizer $optimizer

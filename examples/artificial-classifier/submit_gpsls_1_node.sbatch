#!/bin/bash

## Change this number to increase your parallelism by using more nodes of your
## cluster.
#SBATCH --nodes=1
##
## These seem to be the correct settings for SLURM and Ray to work properly
## together. To actually limit the number of CPU's assigned to each task
## set cpus_per_trial in your scenario file (or on the command line call that
## is performed below.) Ray appears to ignore this number and will always
## attempt to use all available CPUs on a node. To use less than all of the
## CPUs, I have simply lied to ray about how many CPUs are needed per trial.
## Sequential behaviour can be obtained by specifying 1 node with more than
##  half of it's CPUs per trial.
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
##
## Unfortunately, ray doesn't seem to work (reliably) when multiple independent
## runs are performed in parallel on the same node. So I recommend to always 
## ensure that a single instance is run on a node at at ime.
#SBATCH --exclusive
##
## Adjust these as needed for your setup.
#SBATCH -p ada_cpu_short
#SBATCH --mem-per-cpu=3000MB
#SBATCH --time=0-3
#SBATCH --job-name=cqa_ac
#SBATCH --output=cqa_%a.log
##
## Each entry in the array will run the configuration procedure one time. Thus,
## this will perform 3 independent runs of the configurator.
#SBATCH --array=1-3


# This will vary depending on your environment
export PATH="/global/scratch/ypushak/anaconda3/envs/ray-env/bin:$PATH"
# Optionally direct where ray creates files it uses internally
HOME="/global/scratch/ypushak/ray"


# I typically always leave the following code block in here, regardless of the
# number of nodes, since it checks the number of available nodes and sets up
# the environment according to whether or not you are running on more than one
# or only one.
echo $SLURM_NTASKS
let "worker_num=(${SLURM_NTASKS} - 1)"

if [ $SLURM_NTASKS -eq 1 ]
then
    echo "Running on a single node."
    ip_head='None'
else
    echo "Setting up to run on multiple nodes."
    suffix='6379'
    ip_head=`hostname`:$suffix

    # Start the ray head node on the node that executes this script
    # Have the dashboard listen to 0.0.0.0 to bind it to all
    # network interfaces. This allows to access the dashboard through port-forwarding:
    # Let's say the hostname=cluster-node-500 To view the dashboard on localhost:8265, 
    # set up an ssh-tunnel like this: (assuming the firewall allows it)
    # $  ssh -N -f -L 8265:cluster-node-500:8265 user@big-cluster
    srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=`hostname` ray start --head --block --dashboard-host 0.0.0.0 --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &
    # Make sure the head successfully starts before any worker does, otherwise
    # the worker will not be able to connect to redis. In case of longer delay,
    # adjust the sleeptime above to ensure proper order.
    sleep 5

    # Now we execute worker_num worker nodes on all nodes in the allocation except hostname by
    # specifying --nodes=${worker_num} and --exclude=`hostname`. Use 1 task per node, so worker_num tasks in total
    # (--ntasks=${worker_num}) and 5 CPUs per task (--cps-per-task=${SLURM_CPUS_PER_TASK}).
    srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=`hostname` ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
    sleep 5
fi

# You can also specify scenario file arguments as command line arguments here.
run_directory=$(pwd)
optimizer='GPSLS'
scenario='scenario.txt' 
cpus_per_trial=1

cd ../../

echo "python run.py --run_directory $run_directory --scenario $scenario --optimizer $optimizer --cluster-ip-address $ip_head"
python run.py --run_directory $run_directory --scenario $scenario --optimizer $optimizer --cluster-ip-addres $ip_head

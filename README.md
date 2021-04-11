# Landscape-Aware Automated Algorithm Configuration

Includes two experimental automated algorithm configiruation procedures designed
to exploit landscape structure. In particular, these configurators were designed
to target AutoMl hyper-parameter optimization scenarios.

This repository is a work in progress and it builds upon the line of research:

 - Yasha Pushak and Holger H. Hoos.  
**Golden Parameter Search: Exploiting Structure to Quickly Configure Parameters
In Parallel.**  
*In Proceedings of the Twenty-Second Interntional Genetic and Evolutionary 
Computation Conference (GECCO 2020)*. pp 245-253 (2020).  
**Won the 2020 GECCO ECOM Track best paper award.**
 - Yasha Pushak and Holger H. Hoos.  
**Algorithm Configuration Landscapes: More Benign than Expected?**  
*In Proceedings of the Fifteenth Internationl Conference on Parallel Problem 
Solving from Nature (PPSN 2018)*. pp 271-283 (2018).  
**Won the 2018 PPSN best paper award.**

# Table of Contents

   * [Landscape-Aware Automated Algorithm Configuration](#landscape-aware-automated-algorithm-configuration)
   * [Table of Contents](#table-of-contents)
   * [Installing LAAAC](#installing-laaac)
   * [Quick Start Guide](#quick-start-guide)
   * [LAAAC Arguments](#laaac-arguments)
      * [Setup Arguments](#setup-arguments)
      * [Scenario Arguments](#scenario-arguments)
   * [Contact](#contact)

# Installing LAAAC

 1. Create a python virtual environment
 4. Download the latest version of LAAAC from https://github.com/YashaPushak/LAAAC
 5. While in the main LAAAC directory, install LAAAC's other required python 
packages
`pip install -r requirements.txt`.
 6. While in the main LAAAC directory, install LAAAC with 
`pip install .`

# Quick Start Guide

See `./examples/artificial-classifier/` for examples on how to use LAAAC. In particular,
you will need to implement a python interface for a target algorithm runner like the one in
`./examples/artificial-classifier/algorithm_runner.py`. 

If you want to run LAAAC locally from the command line, you can see how to do so by looking at the files:
`./examples/artificial-classifier/run_gpsls_locally.sh` and `./examples/artificial-classifier/run_cqa_locally.sh`.
Note that both of those examples will parallelize the evaluation of the target algorithm on all available
local CPUs. To take advantage of a SLURM cluster, see the examples 
`./examples/artificial-classifier/submit_cqa_3_nodes.sbatch` and
`./examples/artificial-classifier/submit_gpsls_1_node.sbatch`.

The results of all of the runs will be saved in `<experiment_directory>/laaac-output/`. You should find
a directory for each optimizer (that has been used), log files, incumbent trajectories (as a csv file)
and the raw ray.tune output files. If a run crashes and you need to extract the anytime incumbent
trajectories from the raw files, see `./examples/artificial-classifier/get_results_from_logs.py`. Note
that you can run this as a regular file, or as a streamlit application using 
`pip install streamlit`
and then
` streamlit run get_results_from_logs.py`


# LAAAC Arguments

The following is a summary of the command-line arguments exposed by LAAAC at this time. Many additional
parameters are not current exposed as arguments (in particular, the parameters of the optimizers). To 
modify those you will need to modify the file `run.py` for now.


## Setup Arguments

These are general optimizer arguments that are used to set up the optimizer run.

### experiment_dir

<table>
<tr><td><b>Description</b></td><td>The root directory from which experiments will be run. By default, this is the current working directory. The optimizer will change to this directory prior to running, this means that if relative paths are specified for any other files or directories then they must be given relative to your experiment directory.</td></tr>
<tr><td><b>Default</b></td><td><code>.</code></td></tr>
<tr><td><b>Aliases</b></td><td><code>--experiment-dir</code>, <code>--experiment_dir</code>, <code>--experimentDir</code>, <code>--experimentdir</code>, <code>--exec-dir</code>, <code>--exec_dir</code>, <code>--execDir</code>, <code>--execdir</code>, <code>-e</code>, <code>--run-directory</code>, <code>--run_directory</code>, <code>--runDirectory</code>, <code>--rundirectory</code></td></tr>
</table>

### optimizer

<table>
<tr><td><b>Description</b></td><td>Determines what kind of optimizer is used to explore the configuration space. Available options are: "GPS", which uses a golden-section-search-based line-search procedure along randomly chosen vectors of the configuration space; "CQA", which sequentially fits convex quadratic approximations of the loss landscape that under-estimate observed losses and uses the resulting models to guide the search process.</td></tr>
<tr><td><b>Default</b></td><td>GPSLS</td></tr>
<tr><td><b>Aliases</b></td><td><code>--optimizer</code></td></tr>
</table>

### output_dir

<table>
<tr><td><b>Description</b></td><td>The directory where output will be stored. The actual directory for a particular optimizer run with run ID run_id will be stored in {experiment-dir}/{output-dir}/{optimizer}/{run_id}</td></tr>
<tr><td><b>Default</b></td><td><code>laaac-output</code></td></tr>
<tr><td><b>Aliases</b></td><td><code>--output-dir</code>, <code>--output_dir</code>, <code>--outputDir</code>, <code>--outputdir</code>, <code>--output-directory</code>, <code>--output_directory</code>, <code>--outputDirectory</code>, <code>--outputdirectory</code>, <code>--out-dir</code>, <code>--out_dir</code>, <code>--outDir</code>, <code>--outdir</code>, <code>--log-location</code>, <code>--log_location</code>, <code>--logLocation</code>, <code>--loglocation</code></td></tr>
</table>

### scenario_file

<table>
<tr><td><b>Description</b></td><td>The scenario file (and location) that defines what settings are used for optimizer.</td></tr>
<tr><td><b>Default</b></td><td>None</td></tr>
<tr><td><b>Aliases</b></td><td><code>--scenario-file</code>, <code>--scenario_file</code>, <code>--scenarioFile</code>, <code>--scenariofile</code>, <code>--scenario</code></td></tr>
</table>

### verbose

<table>
<tr><td><b>Description</b></td><td>Controls the verbosity of the output. Set of 0 for warnings only. Set to 1 for more informative messages. And set to 2 for debug-level messages. The default is 1.</td></tr>
<tr><td><b>Default</b></td><td>1</td></tr>
<tr><td><b>Aliases</b></td><td><code>--verbose</code>, <code>--verbosity</code>, <code>--log-level</code>, <code>--log_level</code>, <code>--logLevel</code>, <code>--loglevel</code>, <code>-v</code></td></tr>
</table>

## Scenario Arguments

These arguments define the scenario-specific information.

### algo

<table>
<tr><td><b>Description</b></td><td>This should be the name of the python file that implements the target-algorithm interface.</td></tr>
<tr><td><b>Required</b></td><td>Yes</td></tr>
<tr><td><b>Aliases</b></td><td><code>--algo</code>, <code>--algo-exec</code>, <code>--algo_exec</code>, <code>--algoExec</code>, <code>--algoexec</code>, <code>--algorithm</code>, <code>--wrapper</code></td></tr>
</table>

### algo_cutoff_time

<table>
<tr><td><b>Description</b></td><td>The wallclock time limit for an individual target algorithm run, in seconds. If set to zero, no cutoff time will be used.</td></tr>
<tr><td><b>Default</b></td><td>0.0</td></tr>
<tr><td><b>Aliases</b></td><td><code>--algo-cutoff-time</code>, <code>--algo_cutoff_time</code>, <code>--algoCutoffTime</code>, <code>--algocutofftime</code>, <code>--target-run-cputime-limit</code>, <code>--target_run_cputime_limit</code>, <code>--targetRunCputimeLimit</code>, <code>--targetruncputimelimit</code>, <code>--cutoff-time</code>, <code>--cutoff_time</code>, <code>--cutoffTime</code>, <code>--cutofftime</code>, <code>--cutoff</code></td></tr>
</table>

### cluster_ip_address

<table>
<tr><td><b>Description</b></td><td>Can be used to specify the IP address of a ray-cluster to connect to and use for parallelism.set to None (the default) to run locally. Note that the ray-cluster must already be running and configured for this to work if not None.</td></tr>
<tr><td><b>Default</b></td><td>None</td></tr>
<tr><td><b>Aliases</b></td><td><code>--cluster-ip-address</code>, <code>--cluster_ip_address</code>, <code>--clusterIpAddress</code>, <code>--clusteripaddress</code>, <code>--cluster</code></td></tr>
</table>

### config_budget

<table>
<tr><td><b>Description</b></td><td>Limits the sum of the fidelity budget used to evaluate the configurations. For backwards compatability, this contains aliases for runcount limit, for which the name only makes sense if lower fidelity evaluations of a configuration correspond to fewer independent runs of the target algorithm. In practise, if the measure of fidelity corresponds to, <i>e.g.</i>, training iterations of an algorithm, then use this setting to limit the sum of the training iterations across all independent runs of the target algorithm. Either this or the wallclock limit must be less than the maximum integer value. The default is the maximum integer value.</td></tr>
<tr><td><b>Default</b></td><td>2147483647</td></tr>
<tr><td><b>Aliases</b></td><td><code>--config-budget</code>, <code>--config_budget</code>, <code>--configBudget</code>, <code>--configbudget</code>, <code>--configuration-fidelity-budget</code>, <code>--configuration_fidelity_budget</code>, <code>--configurationFidelityBudget</code>, <code>--configurationfidelitybudget</code>, <code>--runcount-limit</code>, <code>--runcount_limit</code>, <code>--runcountLimit</code>, <code>--runcountlimit</code>, <code>--total-num-runs-limit</code>, <code>--total_num_runs_limit</code>, <code>--totalNumRunsLimit</code>, <code>--totalnumrunslimit</code>, <code>--num-runs-limit</code>, <code>--num_runs_limit</code>, <code>--numRunsLimit</code>, <code>--numrunslimit</code>, <code>--number-of-runs-limit</code>, <code>--number_of_runs_limit</code>, <code>--numberOfRunsLimit</code>, <code>--numberofrunslimit</code></td></tr>
</table>

### cpus_per_trial

<table>
<tr><td><b>Description</b></td><td>The number of CPUs that will be allocated to each target algorithm run.</td></tr>
<tr><td><b>Default</b></td><td>1</td></tr>
<tr><td><b>Aliases</b></td><td><code>--cpus-per-trial</code>, <code>--cpus_per_trial</code>, <code>--cpusPerTrial</code>, <code>--cpuspertrial</code></td></tr>
</table>

### crashed_loss

<table>
<tr><td><b>Description</b></td><td>The loss or solution quality score to assign to configurations which do not properly terminate or report valid scores. This should ideally be larger than the largest value your algorithm can return for the loss. If the optimizer is CQA, then this must be finite, and ideally should not be more than one order of magnitude other than the largest value return by your algorithm or any crashed results may substantially biase the fitted models.</td></tr>
<tr><td><b>Required</b></td><td>Yes</td></tr>
<tr><td><b>Aliases</b></td><td><code>--crashed-loss</code>, <code>--crashed_loss</code>, <code>--crashedLoss</code>, <code>--crashedloss</code>, <code>--crashed-score</code>, <code>--crashed_score</code>, <code>--crashedScore</code>, <code>--crashedscore</code>, <code>--crash-score</code>, <code>--crash_score</code>, <code>--crashScore</code>, <code>--crashscore</code>, <code>--crash-loss</code>, <code>--crash_loss</code>, <code>--crashLoss</code>, <code>--crashloss</code></td></tr>
</table>

### fidelity_step

<table>
<tr><td><b>Description</b></td><td>The amount by which the fidelity of the budget should be incremented at each call to the target algorithm runner. Ignored if objective is not "final".</td></tr>
<tr><td><b>Default</b></td><td>1.0</td></tr>
<tr><td><b>Aliases</b></td><td><code>--fidelity-step</code>, <code>--fidelity_step</code>, <code>--fidelityStep</code>, <code>--fidelitystep</code>, <code>--budget-step</code>, <code>--budget_step</code>, <code>--budgetStep</code>, <code>--budgetstep</code>, <code>--fidelity-budget-step</code>, <code>--fidelity_budget_step</code>, <code>--fidelityBudgetStep</code>, <code>--fidelitybudgetstep</code></td></tr>
</table>

### gpus_per_trial

<table>
<tr><td><b>Description</b></td><td>The number of GPUs that will be allocated to each target algorithm run.</td></tr>
<tr><td><b>Default</b></td><td>0</td></tr>
<tr><td><b>Aliases</b></td><td><code>--gpus-per-trial</code>, <code>--gpus_per_trial</code>, <code>--gpusPerTrial</code>, <code>--gpuspertrial</code></td></tr>
</table>

### grace_period

<table>
<tr><td><b>Description</b></td><td>The minimum fidelity budget that a configuration must be evaluated on before it can be eliminated.</td></tr>
<tr><td><b>Default</b></td><td>1.0</td></tr>
<tr><td><b>Aliases</b></td><td><code>--grace-period</code>, <code>--grace_period</code>, <code>--gracePeriod</code>, <code>--graceperiod</code>, <code>--min-fidelity</code>, <code>--min_fidelity</code>, <code>--minFidelity</code>, <code>--minfidelity</code>, <code>--min-budget</code>, <code>--min_budget</code>, <code>--minBudget</code>, <code>--minbudget</code>, <code>--minimum-runs</code>, <code>--minimum_runs</code>, <code>--minimumRuns</code>, <code>--minimumruns</code>, <code>--min-runs</code>, <code>--min_runs</code>, <code>--minRuns</code>, <code>--minruns</code></td></tr>
</table>

### instance_file

<table>
<tr><td><b>Description</b></td><td>The file (and location) containing the names of the instances to be used to evaluate the target algorithm's configurations. If no instances are to be used, set to "None".</td></tr>
<tr><td><b>Required</b></td><td>Yes</td></tr>
<tr><td><b>Aliases</b></td><td><code>--instance-file</code>, <code>--instance_file</code>, <code>--instanceFile</code>, <code>--instancefile</code>, <code>--instances</code>, <code>-i</code></td></tr>
</table>

### integer_fidelity

<table>
<tr><td><b>Description</b></td><td>If True, the fidelity budget will be rounded and cast to an integer after incrementing it by the budget step.</td></tr>
<tr><td><b>Default</b></td><td>True</td></tr>
<tr><td><b>Aliases</b></td><td><code>--integer-fidelity</code>, <code>--integer_fidelity</code>, <code>--integerFidelity</code>, <code>--integerfidelity</code>, <code>--integer-fidelity-budget</code>, <code>--integer_fidelity_budget</code>, <code>--integerFidelityBudget</code>, <code>--integerfidelitybudget</code>, <code>--integer-budget</code>, <code>--integer_budget</code>, <code>--integerBudget</code>, <code>--integerbudget</code></td></tr>
</table>

### max_fidelity

<table>
<tr><td><b>Description</b></td><td>The largest fidelity budget with which to evaluate any single configuration.</td></tr>
<tr><td><b>Default</b></td><td>100.0</td></tr>
<tr><td><b>Aliases</b></td><td><code>--max-fidelity</code>, <code>--max_fidelity</code>, <code>--maxFidelity</code>, <code>--maxfidelity</code>, <code>--max-t</code>, <code>--max_t</code>, <code>--maxT</code>, <code>--maxt</code>, <code>--max-runs</code>, <code>--max_runs</code>, <code>--maxRuns</code>, <code>--maxruns</code></td></tr>
</table>

### objective

<table>
<tr><td><b>Description</b></td><td>Must be one of "mean" or "final". If "mean", the optimizer will seek to optimize the mean loss reported for each fidelity budget. This should always be preferred when the measure of fidelity corresponds to, <i>e.g.</i>, the number of independent runs of the target algorithm on different cross-validation folds or problem instances. However, if the measure of fidelity corresponds to target algorithm training iterations, then "final" should be preferred.</td></tr>
<tr><td><b>Default</b></td><td>mean</td></tr>
<tr><td><b>Aliases</b></td><td><code>--objective</code></td></tr>
</table>

### pcs_file

<table>
<tr><td><b>Description</b></td><td>The file that contains the algorithm parameter configuration space in PCS format. The optimizers support the same syntax used for BOHB, SMAC and ParamILS.</td></tr>
<tr><td><b>Required</b></td><td>Yes</td></tr>
<tr><td><b>Aliases</b></td><td><code>--pcs-file</code>, <code>--pcs_file</code>, <code>--pcsFile</code>, <code>--pcsfile</code>, <code>--param-file</code>, <code>--param_file</code>, <code>--paramFile</code>, <code>--paramfile</code>, <code>--p</code></td></tr>
</table>

### wallclock_limit

<table>
<tr><td><b>Description</b></td><td>Limits the total wall-clock time used by the optimizer, in seconds. Either this, the runcount  or the CPU time limit must be less than the maximum integer value. The default is the maximum integer value.</td></tr>
<tr><td><b>Default</b></td><td>2147483647.0</td></tr>
<tr><td><b>Aliases</b></td><td><code>--wallclock-limit</code>, <code>--wallclock_limit</code>, <code>--wallclockLimit</code>, <code>--wallclocklimit</code>, <code>--runtime-limit</code>, <code>--runtime_limit</code>, <code>--runtimeLimit</code>, <code>--runtimelimit</code></td></tr>
</table>

# Contact

Yasha Pushak  
ypushak@cs.ubc.ca  

PhD Student & Vanier Scholar  
Department of Computer Science  
The University of British Columbia  

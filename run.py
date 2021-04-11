#!/usr/bin/env python3

import copy
import time
import importlib
import logging 

import numpy as np
import pandas as pd

import ray
from ray.tune import run

from ConfigSpace.read_and_write import pcs_new as pcs

from laaac import helper
from laaac.helper import get_configuration_space
from laaac.helper import cd
from laaac.stopper import BudgetStopper
from laaac.nonresumable_trainable import nonresumable_trainable_factory
from laaac.gps_trainable import gps_trainable_factory
from laaac.cqa_searcher import CQASearcher
from laaac import gps_ls_factory
from laaac import BFASHA
from laaac import args
from laaac import results


argument_parser = args.ArgumentParser()
arguments, skipped_lines = argument_parser.parse_arguments()

try:
    with cd(arguments['experiment_dir']):
        run_id = helper.generateID()
        log_location = f'{arguments["output_dir"]}/{arguments["optimizer"].upper()}'
        log_file = f'{arguments["optimizer"].upper()}_{run_id}.log'
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
        helper.mkdir(log_location)
        logger = logging.getLogger('LAAAC')
        fh = logging.FileHandler(f'{log_location}/{log_file}')
        fh.setFormatter(formatter)
        log_map = {0: logging.WARNING,
                   1: logging.INFO,
                   2: logging.DEBUG}
        logging.basicConfig(level=log_map[arguments['verbose']])
        fh.setLevel(log_map[arguments['verbose']])
        logger.addHandler(fh)
        
        if not arguments['integer_fidelity']:
            budget_transformer = lambda b: b*arguments['fidelity_step']
        else:
            budget_transformer = lambda b: int(np.round(b*arguments['fidelity_step']))
    
        cluster_ip = arguments['cluster_ip_address']
        if cluster_ip != 'None':
            logger.info('Setting up ray to run on a cluster')
            ray.init(address=cluster_ip)
            logger.info('Nodes in the Ray cluster:')
            logger.info(ray.nodes())
    
        objective_is_mean = arguments['objective'].lower() == 'mean'
        # asynchronous hyperband early stopping, configured with
        asha_kwargs = dict(
            time_attr='number_of_step_calls',
            mode='min',
            grace_period=int(np.round(arguments['grace_period']/arguments['fidelity_step'])),
            max_t=int(np.round(arguments['max_fidelity']/arguments['fidelity_step'])),
            metric='mean_loss' if objective_is_mean else 'current_loss')
        logger.info('BF-ASHA key-word arguments:')
        logger.info(asha_kwargs)
    
        with open(arguments['pcs_file']) as f_in:
            config_space = pcs.read(f_in)
    
        if arguments['optimizer'].lower() == 'cqa':
            searcher_kwargs = {
                'config_space': config_space,
                'budget': 'number_of_step_calls' if objective_is_mean else 'current_fidelity_budget',
                'failed_result': arguments['crashed_loss'],
                'wall_time': 'current_runtime',
                'wall_time_budget': arguments['wallclock_limit'] if arguments['wallclock_limit'] < 2147483647 else None,
                'log_location': f'{log_location}/{log_file}',
                'log_level': log_map[arguments['verbose']],
                'log_formatter': formatter}
            searcher_kwargs.update(asha_kwargs)
            del searcher_kwargs['time_attr']
            logger.info('CQA Searcher key-word arguments:')
            logger.info(searcher_kwargs)
            searcher = CQASearcher(**searcher_kwargs)
            search_kwargs = {'search_alg': searcher}
        else:
            # Maps the ConfigSpace object into ray's format.
            configuration_space = get_configuration_space(config_space)
            search_kwargs = {'config': configuration_space}
    
        algorithm = arguments['algo']
        with helper.dir_in_path('/'.join(algorithm.split('/')[:-1])):
            module_name = algorithm.split('/')[-1][:-3]
            target_runner = importlib.import_module(module_name)
    
        # Create a trainable class that wraps GPS's class API
        GPSTrainable = gps_trainable_factory(
            target_runner.TargetAlgorithmRunner, config_space, 
            cutoff_time=arguments['algo_cutoff_time'] if arguments['algo_cutoff_time'] > 0 else None,
            instances=helper.read_instances(arguments['instance_file']), 
            wallclock_limit=arguments['wallclock_limit'],
            crash_score=arguments['crashed_loss'],
            catch_exceptions=True)
    
        stopper_kwargs = {'config_budget': arguments['config_budget'],
                          'log_location': f'{log_location}/{log_file}',
                          'log_level': log_map[arguments['verbose']],
                          'log_formatter': formatter}
        if not objective_is_mean:
            reduction_factor = 4 if arguments['optimizer'] in ['CQA'] else 2
            # TODO: Continue from here.
            # Create a non-resumable trainable class that wraps the GPS trainable
            GPSTrainable = nonresumable_trainable_factory(GPSTrainable, budget_transformer=budget_transformer, reduction_factor=reduction_factor, 
                                                          **asha_kwargs)
            # GPS-ASHA doesn't need to modify the stopper to check if the data is fresh
            # because GPS-ASHA already internally checks to see if the data is fresh and
            # only records its budget content if it is. 
            if args.method in ['CQA']:
                stopper_kwargs['count_if_true_column'] = 'fresh'
        logger.info('Stopper key-word arguments:')
        logger.info(stopper_kwargs)    
    
    
        # Create a trainable class that performs a GPS line search on a tune's trainable class API.
        gps_ls_kwargs = copy.deepcopy(asha_kwargs)
        del gps_ls_kwargs['metric']
        TrainableLS = gps_ls_factory(GPSTrainable, config_space,
                                     metric='current_loss',  # Always use the current loss because GPS-LS well take the mean of it. 
                                     objective='mean' if objective_is_mean else 'final',
                                     **gps_ls_kwargs)
   
        if arguments['optimizer'] in ['GPSLS']: 
            FinalTrainable = TrainableLS
            # GPS-LS performs several trail steps per step, so we need to use
            # a column created by GPS-LS that tracks a list of only those 
            # fidelity budgets that have actually been used.
            stopper_kwargs['column_name'] = 'new_current_fidelity_budget'
        else:
            FinalTrainable = GPSTrainable
            stopper_kwargs['column_name'] = 'current_fidelity_budget'

        FinalTrainable = GPSTrainable if arguments['optimizer'] in ['CQA'] else TrainableLS
        budget_stopper = BudgetStopper(**stopper_kwargs)   

        retries = 0
        run_result = None
        while run_result is None:
            try:
                # Use an ASHA scheduler to dynamically stop poorly performing GPS line searchers.
                run_kwargs = dict(
                    run_or_experiment=FinalTrainable,
                    name=run_id,
                    local_dir=log_location,
                    scheduler = BFASHA(**asha_kwargs),
                    time_budget_s=arguments['wallclock_limit'],
                    num_samples=10000,
                    resources_per_trial={
                        'cpu': arguments['cpus_per_trial'],
                        'gpu': arguments['gpus_per_trial']
                    },
                    stop=budget_stopper,
                    **search_kwargs)
                logger.info('ray.tune.run key-word arguments:')
                logger.info(run_kwargs)
                run_result = run(**run_kwargs)
            except RuntimeError as e:
                # Sometimes ASHA raises a RuntimeError because the previous redis connection is still
                # shutting down and is not yet ready to accept a new one. 
                retries += 1
                if retries >= 10:
                    raise
                logger.warning('Caught a RuntimeError while trying to start the configuration run.', exc_info=e)
                logger.info(f'The redis server may not have been ready for a connection yet. '
                             f'Sleeping for 1 minute and trying {10-retries} more times...')
                time.sleep(60)
    
        logger.info('{}+BF-ASHA {} Run complete.'.format(arguments['optimizer'], run_id))
    
        # If the run crashes or runs out of time, you can still perform this call on the results
        # that managed to be completed by reading in the run result Analysis. Look up how to
        # get ray.tune Analysis objects from the logged results and then call this function.
        incumbent_trajectory, final_results = results.analyze_results(
            run_result, objective=arguments['objective'], gpsls=arguments['optimizer'].lower() == 'gpsls')
       
        logger.info('Final results:')
        logger.info(final_results)
        with pd.option_context('display.max_rows', 50, 'display.max_columns', 10):
            logger.info(f'Incumbent Trajectory:\n{incumbent_trajectory}')
        incumbent_trajectory.to_csv(f'{log_location}/incumbent_trajectory_{run_id}.csv')
finally:
    ray.shutdown()

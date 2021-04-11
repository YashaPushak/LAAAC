import copy
import os
import time
import sys
import json

import numpy as np

from ray.tune import Trainable

from ConfigSpace.util import deactivate_inactive_hyperparameters

from laaac import helper

def gps_trainable_factory(TargetAlgorithmRunner, config_space, cutoff_time=None, instances=None,
                          wallclock_limit=None, crash_score=1, catch_exceptions=True, **kwargs):
    """gps_trainable_factory

    GPS ASHA requires various results to have been returned by the trainable.
    As a convenience method, this will extract all such necessary metadata (and
    some extras) from a TargetAlgorithmRunner originally designed for use with
    GPS.

    Parameters
    ----------
    TargetAlgorithmRunner : <class 'GPS.abstract_runner.AbstractRunner'>
      An class that implements the GPS.abstract_runner.AbstractRunnner
      interface.
    config_space : ConfigSpace.configuration_space.ConfigurationSpace
      The configuration space of the target algorithm runner.
    cutoff_time : float | None
      If specified, the target algorithm runner will attempt to enforce a
      wall-clock time on the individual target algorithm runs (individual
      calls to step).
    instances : list | None
      A list of instances (possibly strings of filenames) from which a random
      value will be passed to your target algorithm runner in each step. If
      None, then None will be passed as the only instance to your target
      algorithm runner.
    wallclock_limit : float | None
      If not None, new target algorithm runs will not be performed after
      this many seconds have elapsed since the instantiation of the GPS
      trainable. This is a convenience function for helping to enforce wall
      clock time limits on the overall configuration process.
    crash_score : float
      Used to assign a loss/score to runs of the target algorithm runner that
      crash or are not run due to the wallclock limit having been exhausted.
    catch_exceptions : bool
      By default, GPSTrainables will catch exceptions raised by your target
      algorithm runner and assign them with a loss using the crash score.
      However, this behaviour can be overridden by setting this to True. This
      is mostly useful for debugging.
    kwargs : dict
      Aditional key-work arguments that should be passed to your
      TargetAlgorithmRunner class when it is instantiated.
    
    Returns
    -------
    <class 'GPSTrainable'>
      A class which can be instantiated to create a GPSTrainable object.
    """
    if catch_exceptions:
        ExceptionsToCatch = Exception
    else:
        class DummyException(Exception):
            pass
        ExceptionsToCatch = DummyException
    # Rename to avoid scope collision
    tar_kwargs = kwargs


    class GPSTrainable(Trainable):
    
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.crash_score = crash_score
            self.training_budget = 1
            self.target_runner = TargetAlgorithmRunner(**tar_kwargs)
            # Get the cutoff time for the runs
            self.cutoff = cutoff_time
            self.wallclock_limit = np.inf if wallclock_limit is None else wallclock_limit
            # Get the set of instances (train/test split seeds) to use for the runs
            self.instances = instances if isinstance(instances, list) else [None]
            self.start_time = time.time()
    
        def setup(self, config):
            self.timestep = 0
            self.runtimes = []
            self.results = []
            self.losses = []
            self.messages = []
            self.timestamps = []
            self.training_budgets = []
    
        def save_checkpoint(self, checkpoint_dir, checkpoint_dict=None):
            data = {'timestep': self.timestep,
                    'runtimes': self.runtimes,
                    'results': self.results,
                    'losses': self.losses,
                    'messages': self.messages,
                    'timestamps': self.timestamps,
                    'training_budgets': self.training_budgets}
            path = os.path.join(checkpoint_dir, 'checkpoint')
            if checkpoint_dict is None:
                with open(path, 'w') as f:
                    f.write(json.dumps(data))
            else:
                checkpoint_dict[path] = data
            return path
    
        def load_checkpoint(self, checkpoint_path, checkpoint_dict=None):
            if checkpoint_dict is None:
                with open(checkpoint_path) as f:
                    data = json.loads(f.read())
            else:
                data = checkpoint_dict[checkpoint_path]
            self.timestep = data['timestep']
            self.runtimes = data['runtimes']
            self.results = data['results']
            self.losses = data['losses']
            self.messages = data['messages']
            self.timestamps = data['timestamps']
            self.training_budgets = data['training_budgets']
           
        def step(self):
            """step
    
            Performs a single run/iteration of the target algorithm's training
            procedure using self.config.
            """      
            start = time.time()
            try:
                if time.time() < self.wallclock_limit + self.start_time:
                    config = deactivate_inactive_hyperparameters(copy.deepcopy(self.config), config_space)
                    config = dict(config)
                    result, runtime, loss, misc \
                        = self.target_runner.perform_run(parameters=config, 
                                                         instance=np.random.choice(self.instances),
                                                         seed=np.random.randint(10000, 99999),
                                                         cutoff=self.cutoff,
                                                         run_length=self.training_budget,
                                                         run_id=helper.generateID())
                    # Sometimes nans are returned by some algorithms. These should be treated as crashed runs
                    loss = self.crash_score if np.isnan(loss) else loss
                else:
                    # The overall configuration wall clock budget has been exhausted. So
                    # we are now going to skip this and all future target algorithm runs
                    result = 'SKIPPED'
                    runtime = 0
                    loss = self.crash_score
                    misc = 'Skipped because the wall clock configuration budget was exhausted'
            except ExceptionsToCatch as e:
                # The run crashed
                result = 'CRASHED'
                runtime = time.time() - start
                loss = self.crash_score
                misc = ('The run crashed with an exception of type {} and message {}'
                        ''.format(type(e), repr(e)))
            # Record the start and end timestamps and the message
            end_timestamp = time.time()
            self.timestamps.append((start, end_timestamp))
            self.messages.append(misc)
            self.results.append(result)
            self.runtimes.append(runtime)
            self.losses.append(loss)
            self.training_budgets.append(self.training_budget)
            result = {'mean_loss': float(np.mean(self.losses)),
                      'current_loss': loss,
                      'all_losses': self.losses,
                      'number_of_step_calls': len(self.losses),
                      'current_fidelity_budget': self.training_budget,  # The amount just spent
                      'cumulative_fidelity_budget': np.sum(self.training_budgets),  # The sum of all fidelity budgets spent
                      'all_fidelity_budgets': self.training_budgets,  # The fidelity budgets spent at each call to step
                      'current_runtime': runtime,
                      'total_runtimes': np.sum(self.runtimes),
                      'all_runtimes': self.runtimes,
                      'current_end_timestamp': end_timestamp,
                      'all_timestamps': self.timestamps,
                      'current_message': misc,
                      'all_messages': self.messages,
                      'current_result': result,
                      'all_results': self.results,
                      'configuration': str(self.config)}
            return result
                   
    return GPSTrainable

def _to_type(list_, type_):
    return [type_(element) for element in list_]

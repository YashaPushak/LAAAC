import pickle
import copy
import os
import string
import time
import logging

import numpy as np
import pandas as pd
from scipy import stats

import ConfigSpace
from ConfigSpace.read_and_write import pcs_new as pcs
from ConfigSpace.hyperparameters import UniformFloatHyperparameter as Float
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as Integer

def gps_ls_factory(UserTrainable, config_space, metric, mode, max_t, objective, 
                   grace_period=1, reduction_factor=2,
                   elimination_method='t-test', 
                   local_dir='./gps_ls_results',
                   save_trajectory=False,
                   create_checkpoint_directories=False,
                   track_columns=('current_runtime', 'current_fidelity_budget'),
                   end_timestamp_name='current_end_timestamp',
                   *args, **kwargs):

    GR = (1 + np.sqrt(5))/2

    if isinstance(track_columns, str):
        track_columns = tuple([track_columns])
    if track_columns is None:
        track_columns = tuple([])

    if mode not in ['min', 'max']:
        raise ValueError('The mode must be "min" on "max". '
                         'Provided {}.'.format(mode))
    if objective not in ['mean', 'final']:
        raise ValueError('The objective must be "mean" or "final", which '
                         'correspond to optimizing for the mean metric after '
                         'each step vs the final metric value observed. '
                         'Provided {}.'.format(objective))

    if create_checkpoint_directories:
        local_dir = '{}/{}'.format(os.getcwd(), local_dir)
        for i in range(len(local_dir.split('/'))+1):
            try:
                dir_ = '/'.join(local_dir.split('/')[:i])
                os.mkdir(dir_)
            except:
                pass


    class GPSLSTrainable(UserTrainable):
        """GPSLSTrainable

        This is an extra layer designed to sit between ASHA and your Trainable
        class. Instead of simply evaluating the single configuration ASHA requests
        it to evaluate, it will actually always evaluate a pair of configuraitons.
        As far as ASHA is concerned, the performance of the better configuration
        will be the only one that is ever returned. 
        """
 
        def __init__(self, *args, **kwargs):
            self._reserved_startings = ['_bracket_', '_numeric_']
            self._metric = metric
            self._mode = mode
            self._max_t = max_t
            self._grace_period = grace_period
            self._reduction_factor = reduction_factor
            self._local_dir = local_dir
            self._config_space = config_space
            self._mode_op = -1 if self._mode == 'max' else 1

            MAX_RUNGS = int(np.log(max_t/grace_period)/np.log(reduction_factor) + 1)
            self._milestones = np.array([grace_period*reduction_factor**k
                                          for k in reversed(range(MAX_RUNGS))])

            self._objective = objective
            if self._objective == 'mean':
                self._update_eliminated = _t_test
                self._select_incumbent = _select_incumbent_factory(np.mean)
            elif self._objective == 'final':
                self._update_eliminated = _milestone_check
                self._select_incumbent = _select_incumbent_factory(lambda scores: scores[-1])
            else:
                raise ValueError('The elimination method must be "t test", '
                                 '"permutation test" or "milestone". '
                                 'Provided {}.'.format(elimination_method))

            self._save_trajectory = save_trajectory
            if save_trajectory:
                self._bracket_run_id = random_directory(n=6)
                # @TODO: This file location needs to be fixed.
                self._bracket_trajectory_file = './gps_asha_output/{}/incumbent_trajectory.log'.format(self._bracket_run_id)
                log_dir = '/'.join(self._bracket_trajectory_file.split('/')[:-1])
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir) 
                if not os.path.isfile(self._bracket_trajectory_file):
                    with open(self._bracket_trajectory_file, 'w') as f_out:
                        f_out.write(',CPU Time Used,Wallclock Time,Target Algorithm Time,'
                                    '# Target Algorithm Runs,Run ID,Configuration...\n')

            self.config = {}
            UserTrainable.__init__(self, *args, **kwargs)

        def _fit(self, config):
            # Track the numerical hyper-parameters, their bounds and if they 
            # are integers
            hp_names = []
            hp_lowers = []
            hp_uppers = []
            hp_integers = []
            for hp in config:
                current_value = config[hp]
                hp = self._config_space.get_hyperparameter(hp)
                if isinstance(hp, (Float, Integer)):
                    # Sample a new value for this hyperparameter
                    hp_integers.append(isinstance(hp, Integer))
                    hp_names.append(hp.name)
                    hp_lowers.append(hp.lower)
                    hp_uppers.append(hp.upper)
            self._numeric_hp_names = hp_names
            self._numeric_lowers = hp_lowers
            self._numeric_uppers = hp_uppers
            self._numeric_integers = hp_integers

        def _get_random_step(self, config):
            current_values = np.array([config[hp] for hp in self._numeric_hp_names])
            new_values = np.random.triangular(self._numeric_lowers, 
                                              current_values,
                                              self._numeric_uppers)
            self._clip_and_snap(new_values)
            step = new_values - current_values
            return step

        def _clip_and_snap(self, new_values):
            new_values = np.clip(new_values,
                                 self._numeric_lowers,
                                 self._numeric_uppers)
            new_values[self._numeric_integers] = np.array(np.round(new_values[self._numeric_integers]), dtype=int)
            return new_values

        def _apply_step(self, values, step, step_size):
            # values are from the winner
            # Track the new configuration created, with hyper-parameters
            # clipped to min/max values
            new_config = copy.deepcopy(self.config)
            # Track the position of the configuration we would have if there
            # were no bounds or integer values
            #print('Current winner\'s virtual values: {}'.format(values))
            #print('Previous step {}'.format(step))
            #print('Next step size {}'.format(step_size))
            virtual_new_values = values + step*step_size
            # Now clip and snap values.
            new_values = self._clip_and_snap(virtual_new_values)
            #print('New actual values: {}'.format(new_values))
            # Apply the changes to the configuration
            for i in range(len(new_values)):
                new_config[self._numeric_hp_names[i]] = new_values[i] if not self._numeric_integers[i] \
                                                                      else int(new_values[i])
            return new_config, virtual_new_values
            
        def _get_numeric_values(self, config):
            return np.array([config[hp] for hp in self._numeric_hp_names])

        def _take_new_random_step(self, winner, step_size=1):
            # idx is the winner
            config = self._bracket_configs[winner]
            step = self._get_random_step(config)
            #print('New Step: {}'.format(step))
            # We're taking a new step, so discard any old virtual location
            # information and start fresh with the real position of this
            # configuration
            numeric_values = self._get_numeric_values(config)
            self._take_step(winner, numeric_values, step, step_size)
            self._bracket_has_been_worse = [False, False]

        def _take_step(self, winner, numeric_values, step, step_size):
            loser = 1-winner
            # numeric_values are for the winner
            new_config, new_numeric_values = self._apply_step(
                numeric_values, step, step_size)
            logging.debug('New configuration = {}'.format(new_config))
            # Clear the slate
            self._bracket_configs[loser] = new_config
            self._bracket_values[loser] = new_numeric_values
            self._bracket_metrics[loser] = []
            self._bracket_results[loser] = []
            self._bracket_steps[loser] = 0
            self._bracket_eliminated[loser] = False
            # Restart the loser's configuration data
            self._select_bracket_configuration(loser)
            self.config = new_config
            super().setup(new_config)

        def reset_config(self, *args, **kwargs):
            pass # Must be over-written by the user if they need to do anything

        def _get_side(self, step, idx):
            # We need to determine if the configuration idx is on the "left"
            # or the "right" side of the bracket, so that we know if we have
            # eliminated a configuration from both ends. We will arbitrarily
            # assign an ordering based on the ordering of the hyper-parameter
            # with the largest absolute value of the step size
            hp = np.argmax(np.abs(step))
            return 0 if self._bracket_values[idx][hp] < self._bracket_values[1-idx][hp] else 1

        def _take_next_step(self):
            # Get the index of the winner
            winner = 0 if self._bracket_eliminated[1] else 1
            # Make sure the winner is selected so that we dont' end up with
            # a fresh configuration selected.
            self._select_bracket_configuration(winner)
            # Calculate the last step size
            step = self._bracket_values[winner] - self._bracket_values[1-winner]
            was_shrinking = np.all(self._bracket_has_been_worse)
            self._bracket_has_been_worse[self._get_side(step, 1-winner)] = True
            is_shrinking = np.all(self._bracket_has_been_worse)
            if not is_shrinking: 
                # We are growing
                step_size = GR
            elif not was_shrinking: 
                # We were growing, but over-shot
                step_size = -1/GR
            else: 
                # We are shrinking
                step_size = 1/GR
            # Take the step we just calculated
            self._take_step(winner, self._bracket_values[winner], step, step_size)
            # Initialize the new configuration
            self._select_bracket_configuration(1-winner)
            super().setup(self._bracket_configs[1-winner])

        def setup(self, config):
            logging.debug('Setup called with config {}'.format(config))
            # extracts which hyper-parameters are numeric, saves their bounds
            # and whether or not they're integer-valued.
            self._fit(config)
            self._bracket_real_steps = 0
            self._bracket_configs = [config, None]
            self._bracket_values = [self._get_numeric_values(config), None]
            self._bracket_metrics = [[], []]
            self._bracket_results = [[], []]
            self._bracket_steps = [0, 0]
            self._bracket_eliminated = [False, False]
            self._bracket_step_count = 0
            self._bracket_incumbent = 0
            self._bracket_checkpoint_dirs = tuple('{}/{}'.format(self._local_dir,
                                                                 random_directory())
                                                  for _ in range(2))
            # Create the checkpoint directories
            if create_checkpoint_directories:
                for idx in [0, 1]:
                   os.mkdir(self._bracket_checkpoint_dirs[idx])
            self._bracket_checkpoint_paths = [None, None]
            self._bracket_checkpoint_data = {}
            self._bracket_selected = 0
            # Initialize the first configuration's trainable
            self.config = self._bracket_configs[0]
            logging.debug('self.config={}'.format(self.config))
            super().setup(self._bracket_configs[0])
            # Create a challenging configuration by taking a step in a new
            # random direction.
            self._take_new_random_step(0)
            self._bracket_wall_start = time.time()
            self._bracket_cpu_start = time.process_time()
            self._log_incumbent()
            logging.debug('self.config={}'.format(self.config))


           
        def save_checkpoint(self, checkpoint_dir):
            path = os.path.join(checkpoint_dir, 'checkpoint')
            data = {}
            for item in self.__dict__:
                if np.any([item.startswith(start) for start in self._reserved_startings]):
                    data[item] = self.__getattribute__(item)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            return path

        def load_checkpoint(self, checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            for item in data:
                setattr(self, item, data[item])

        def step(self):
            self._bracket_step_count += 1
            restart_loop = True
            count = 0
            tracked_column_values = [[] for _ in range(len(track_columns))]
            tracked_column_timestamps = []
            current_milestones = list(copy.deepcopy(self._milestones))
            while restart_loop:
                restart_loop = False
                # For each step
                for step in range(0, self._bracket_step_count):
                    count += 1
                    changed = False
                    # For each configuration, starting with the one currently loaded
                    for idx in [self._bracket_selected, 1-self._bracket_selected]:
                        # Check if the configuration needs to be evaluated at this
                        # step
                        if (self._bracket_steps[idx] <= step
                            and not self._bracket_eliminated[idx]):
                            changed = True
                            # Evaluate it!
                            self._select_bracket_configuration(idx)
                            result = super().step()
                            self._bracket_real_steps += 1
                            # Only track the budget results if the result is 'fresh'
                            # i.e., the trainable is not a NonResumableTrainable 
                            # that just returned a result already reported at the
                            # last milestone.
                            if result.get('fresh', True):
                                for i in range(len(track_columns)):
                                    tracked_column_values[i].append(result[track_columns[i]])
                                tracked_column_timestamps.append(result[end_timestamp_name] if end_timestamp_name in result else time.time())
                            logging.debug('{stars}step={step}-idx={idx}{stars}'.format(stars='*'*10,
                                                                                       step=step,
                                                                                       idx=idx))
                            logging.debug(result)
                            # Update the results
                            self._bracket_results[idx].append(result)
                            self._bracket_metrics[idx].append(result[self._metric]*self._mode_op)
                            self._bracket_steps[idx] += 1
                    # Check to see if we should eliminate a candidate
                    if changed:
                        self._bracket_eliminated = self._update_eliminated(self._bracket_metrics,
                                                                           milestones=self._milestones,
                                                                           min_steps=self._bracket_step_count/2,
                                                                           current_step=self._bracket_step_count)
                        logging.debug('{0}: Objective: ({1:.4f}, {2:.4f}) - Eliminated? {3}'
                                      ''.format(count,
                                                np.mean(self._bracket_metrics[0]) if mode == 'mean' else self._bracket_metrics[0][-1],
                                                np.mean(self._bracket_metrics[1]) if mode =='mean' else self._bracket_metrics[1][-1],
                                                self._bracket_eliminated))
                    # If someone was eliminated and we're at a milestone
                    if np.any(self._bracket_eliminated) and step in current_milestones:
                        current_milestones.remove(step)
                        # Take a new step
                        self._take_next_step()
                        logging.debug('Taking the next step on this line search')
                        restart_loop = True
                        break 
            # Update the incumbent, if necessary
            self._update_incumbent()
            self.config = self._bracket_configs[self._bracket_incumbent]
            # Return the latest result from the incumbent
            result = copy.deepcopy(self._bracket_results[self._bracket_incumbent][-1])
            result['real_configuration'] = str(self.config)
            result['line_search_end_step_timestamp'] = time.time()
            for i in range(len(track_columns)):
                result['new_{}'.format(track_columns[i])] = tracked_column_values[i]
            result['new_timestamps'] = tracked_column_timestamps
            # This line search is now done, so let's set up for the next one
            #if self._bracket_step_count in self._milestones or True:
            self._take_new_random_step(self._bracket_incumbent, 1/np.sqrt(self._bracket_step_count))
            logging.debug('Line search complete, took a new random step')
            return result
            
        def _update_incumbent(self):
            old_incumbent = self._bracket_incumbent
            self._bracket_incumbent = self._select_incumbent(self._bracket_metrics)
            if old_incumbent != self._bracket_incumbent:
                self._log_incumbent()

        def _log_incumbent(self):
            if self._save_trajectory and len(self._bracket_configs[self._bracket_incumbent]) > 0:
                with open(self._bracket_trajectory_file, 'a') as f_out:
                    f_out.write('{step},{cpu_time},{wall_time},{target_time},{target_runs},{run_id},{config}\n'
                                ''.format(step=self._bracket_step_count,
                                          cpu_time=time.process_time() - self._bracket_cpu_start,
                                          wall_time=time.time() - self._bracket_wall_start,
                                          target_time=-1,
                                          target_runs=self._bracket_real_steps,
                                          run_id=self._bracket_run_id,
                                          config=print_config(self._bracket_configs[self._bracket_incumbent])))
           
        def _select_bracket_configuration(self, idx):
            if idx != self._bracket_selected:
                # Save the current state of the currently selected configuration
                self._bracket_checkpoint_paths[1-idx] = super().save_checkpoint(
                    self._bracket_checkpoint_dirs[1-idx], self._bracket_checkpoint_data)
                # Select the new configuration
                self._bracket_selected = idx
                self.config = self._bracket_configs[idx]
                if self._bracket_checkpoint_paths[idx] is not None:
                    # The current checkpoint path will be none if this is a new
                    # configuration that has never before been saved
                    # Load the new configuration's saved state
                    super().load_checkpoint(self._bracket_checkpoint_paths[idx],
                        self._bracket_checkpoint_data)

        def get_wall_time(self):
            return time.time() - self._bracket_wall_start

        def get_cpu_time(self):
            return time.time(self) - self._bracket_cpu_start

        def get_real_steps(self):
            return self._bracket_real_steps

        def get_incumbent(self):
            return self._bracket_configs[self._bracket_incumbent]

        def get_incumbent_metrics(self):
            return self._bracket_metrics[self._bracket_incumbent]

    return GPSLSTrainable

def _get_n_evaluated(metrics):
    return min(len(metrics[0]), len(metrics[1]))

def random_directory(n=16):
    return ''.join(np.random.choice(list(string.ascii_uppercase + string.digits)) for _ in range(n))

def _t_test(metrics, alpha='adaptive', min_steps=1, current_step=1, **kwargs):
    if alpha == 'adaptive':
        # Starts off with alpha = 0.05 for 2 steps and slowly decreases 
        # for example, alpha = 0.005 at 200 steps.
        alpha = (0.1/np.sqrt(2))/np.sqrt(current_step)
    k = _get_n_evaluated(metrics)
    eliminated = [False, False]
    if k > 1:
        _, p_value = stats.ttest_ind(metrics[0], metrics[1], equal_var=False)
        if p_value < alpha:
            if np.mean(metrics[0]) < np.mean(metrics[1]):
                # Only eliminate a configuration if the new incumbent has been
                # evaluated on at least min_steps.
                if len(metrics[0]) > min_steps:
                    eliminated = [False, True]
            else:
                if len(metrics[1]) > min_steps:
                    eliminated = [True, False]
    return eliminated

def _permutation_test(metrics, alpha=0.05, n_samples=10000, **kwargs):
    raise NotImplementedError('Permutation tests are not yet supported.')  # TODO

def _milestone_check(metrics, milestones=None, **kwargs):
    eliminated = [False, False]
    k = _get_n_evaluated(metrics)
    # Get the largest milestone with at least k steps
    if milestones is not None:
        larger_milestones = np.where(milestones <= k)[0]
        if len(larger_milestones) > 0:
            k = milestones[larger_milestones[0]] - 1
        else:
            k = None
    if k is not None:
        if metrics[0][k] < metrics[1][k]:
            eliminated = [False, True]
        elif metrics[0][k] > metrics[1][k]:
            eliminated = [True, False]
    return eliminated

def _select_incumbent_factory(score):
    def _select_incumbent(metrics):
        # The number of runs for one of the configurations will be larger if
        # the other was eliminated.
        if len(metrics[0]) > len(metrics[1]):
           return 0
        elif len(metrics[1]) > len(metrics[0]):
           return 1
        # They have both been evaluated on an equal number of steps, so we
        # use their actual scores to determine the better one
        k = _get_n_evaluated(metrics)
        if score(metrics[0][:k]) < score(metrics[1][:k]):
            return 0
        elif score(metrics[1][:k]) < score(metrics[0][:k]):
            return 1
        else:
            # Break ties at random
            return np.random.choice([0,1])
    return _select_incumbent

def print_config(config, mode='plain'):
    if mode == 'aclib':
        #print('*'*10 + str(config))
        config_string = ''
        parameters = sorted(list(config.keys()))
        for hp in parameters:
            config_string = "{},{}='{}'".format(config_string, hp, config[hp])
        config_string = config_string[1:]
    else:
        config_string = '"{}"'.format(config)
    return config_string

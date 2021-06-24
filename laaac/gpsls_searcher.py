import time
import copy 
import logging
from collections import defaultdict
from multiprocessing import Process, Manager
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import RobustScaler

try:
    from ray.tune.suggest import Searcher
    ray_available = True
except:
    ray_available = False

from ConfigSpace.hyperparameters import UniformFloatHyperparameter as Float
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as Integer
from ConfigSpace.hyperparameters import CategoricalHyperparameter as Categorical
from ConfigSpace import Configuration

from .cqa import SCQA, GCQA
from .bfasha import BFASHA
from .helper import generateID as generate_id

LOGGER = logging.getLogger('GPS-LS')

GR = (1 + np.sqrt(5))/2


@contextmanager
def exception_logging():
    try:
        yield
    except Exception as e:
        LOGGER.error('An exception occured.', exc_info=e)
        raise e
    finally:
        pass


if ray_available:
    class GPSLSSearcher(Searcher):
        """
        A ray.tune.Searcher wrapper around CoreGPSLSSearcher
        
        See CoreGPSLSSearcher.
        """
        def __init__(self, config_space, metric, *args, mode='min', **kwargs):
            LOGGER.debug('Initializing GPSLSSearcher')
            super().__init__(metric=metric, mode=mode, **kwargs)
            LOGGER.debug('called super().__init__')
            self._gpsls = CoreGPSLSSearcher(config_space, metric, *args, mode=mode, **kwargs)
    
        def suggest(self, *args, **kwargs):
            self._gpsls.suggest(*args, **kwargs)
    
        def on_trial_result(self, *args, **kwargs):
            self._gpsls.on_trial_result(self, *args, **kwargs)
    
        def on_trial_complete(self, *args, **kwargs):
            self._gpsls.on_trial_complete(self, *args, **kwargs) 


class CoreGPSLSSearcher:
    """
    Parameters
    ----------
    config_space : ConfigSpace
      The configuration space.
    metric : str
      The name of the key in the dict of the results from the trainables to be
      optimized.
    mode : 'min' | 'max'
      Whether or not the metric is to be minimized or maximized.
    budget : str
      The name of the key in the dict of the results from the trainable that
      tracks the fidelity/budget upon which the configuration has been
      evaluated so far.
    failed_result : 1
      The value for metric that is to be assumed if the result is not reported,
      is reported as nan, or is reported as non-finite.
    wall_time : str | None
      The name of the key in the dict of the results from the trainable that
      corresponds to the wallclock time cost of having evaluated the
      configuration. This should be non-cumulative over steps. Used to estimate
      how much time is left for the optimization process.
    wall_time_budget : float | None
      The maximum wall time allowed for configuration. Note that the searcher
      does not enforce this as a constraint in any way. Instead, it uses this
      as a means of estimating how much time is left for performing runs.
    log_location : str | None
      Optionally enables logging to a file.
    log_level : int
      A log level, e.g., logging.DEBUG.
    log_formatter : logging.Formatter | str | None
      A log formatter or None to use the default.
    """
    @exception_logging()
    def __init__(self, config_space, metric,  mode='min',
                 budget='budget', failed_result=1,
                 wall_time=None, wall_time_budget=None,
                 log_location='GPSLS.log', log_level=logging.DEBUG,
                 log_formatter='%(asctime)s:%(name)s:%(levelname)s: %(message)s', **kwargs):
        if log_location is not None:
            fh = logging.FileHandler(log_location)
            fh.setLevel(log_level)
            if log_formatter is not None:
                if isinstance(log_formatter, str):
                    log_formatter = logging.Formatter(log_formatter)
                fh.setFormatter(log_formatter)
            LOGGER.addHandler(fh)
        LOGGER.setLevel(log_level)
        LOGGER.debug('**************************************************')
        LOGGER.debug('******** Initializing the GPSLS Searcher *********')
        LOGGER.debug('**************************************************')
        self._config_space = config_space
        self._metric = metric
        self._mode = mode
        self._failed_result = failed_result
        self._budget = budget
        self._wall_time = wall_time
        self._wall_time_budget = wall_time_budget
        self._start_time = time.time()
        self._configurations = {}
        # A dictionary mapping cat_ids to lists of GPS line searches that
        # are being performed over the numeric parameters
        self._gpsls = defaultdict(list)
        # A dictionary mapping config_ids to lists of of tuples of GPS
        # line searches that contain the configuration and the corresponding
        # GPSLS configuration ID. 
        self._trial_id_to_gpsls = defaultdict(list)
        # The incumbents for each set of numeric-valued parameters
        self._num_incumbents = {}
        # A dict mapping trial_ids to dicts mapping fidelity budgets to loss
        self._losses = defaultdict(dict)
        # TODO: Expose parameters
        self._cat_bfasha = BFASHA(grace_period=1, reduction_factor=8)
        self._cat_id_to_key = {}
        self._cat_key_to_id = {}
        self._cat_id_to_config = {}
        self._trial_id_to_cat_num_id = {}
        # Check to see if there are any categorical hyperparameters at all.
        self._cat = False
        for hp in config_space.get_hyperparameters():
            if isinstance(hp, Categorical):
                self._cat = True
                break
        self._transformers = {}
        self._suggested_default = defaultdict(bool)  # initializes to False
        self._suggested_default_cat = False
        LOGGER.debug('**************************************************')
        LOGGER.debug('********* Initialized the GPSLS Searcher *********')
        LOGGER.debug('**************************************************')

    @exception_logging()
    def suggest(self, trial_id):
        LOGGER.debug('********** Suggesting a Configuration **********')
        if self._cat:
            # Together, the cat and num IDs form a unique id from the
            # perspective of categorical BF-ASHA
            cat_id, num_id = self._cat_bfasha.suggest()
            if cat_id not in self._cat_id_to_key:
                LOGGER.debug('BF-ASHA recommends we try evaluating a new set of '
                             'categorical values. Attempting to find one with '
                             'random sampling')
                for _ in range(50):
                    if False and not self._suggested_default_cat:
                        LOGGER.debug('We are starting by suggesting the default configuration '
                                     'for the categorical hyper-parameters')
                        config_obj = self._config_space.get_default_configuration()
                        self._suggested_default_cat = True
                    else:
                        config_obj = self._config_space.sample_configuration()
                    # Convert to a dict to work with internally
                    config = dict(config_obj)
                    # Check to see if this combination of categorical values has been seen before
                    num_hp, num_val, cat_hp, cat_val, cat_key = self._group_hps(config)
                    if cat_key not in self._cat_key_to_id:
                        LOGGER.debug('We found a new set of categorical values to try.')
                        # We found a new categorical combination!
                        # BFASHA assigns its own ids, so create a mapping here.
                        self._cat_id_to_key[cat_id] = cat_key
                        self._cat_key_to_id[cat_key] = cat_id
                        self._cat_id_to_config[cat_id] = config_obj
                        break
                else:
                    # We failed to find a new categorical configuration.
                    # We searched by random sampling so it's possible that
                    # there might still be some out there but we don't want
                    # to waste our time searching for ever, so we'll assume
                    # that there are none. Tell BFASHA to increase its grace
                    # period and give us a new suggestion.
                    LOGGER.debug('After several failed attempts, we did not find any '
                                 'new categorical values to evaluate. So we are telling '
                                 'BF-ASHA not to suggest anything new anymore.')
                    self._cat_bfasha.no_new_suggestions()
                    cat_id, num_id = self._cat_bfasha.suggest()
            # Look up the configuration by its cat id.
            config = self._cat_id_to_config[cat_id]
            self._trial_id_to_cat_num_id[trial_id] = (cat_id, num_id)
        else:
            # There are no categorical hyper-parameters; however, the next
            # part of the code assumes that a configuration has been selected
            # and then it over-writes the numeric-values of the parameters if
            # there are any, so we will simple sample some random values
            config = self._config_space.sample_configuration()
        LOGGER.debug(f'Selected a configuration with some categorical values: {config}')
        # Convert to a dict to work with internally
        config = dict(config)
        num_hp, num_val, cat_hp, cat_val, cat_key = self._group_hps(config)
        # Check to see if new models have been fit.
        if len(num_hp) > 0:
            # Now check to see if this combination of categorical values has any line searchs for
            #  for the numeric hyper-parameters
            np.random.shuffle(self._gpsls[cat_key])
            step_sizes = []
            original_num_val = num_val
            for gpsls in self._gpsls[cat_key]:
                LOGGER.debug('Found a line search for the numeric hyperparameters of the configuration.')
                num_val, config_id = gpsls.suggest()
                if num_val is not None:
                    LOGGER.debug(f'The line search suggested: {num_val}')
                    break
                LOGGER.debug('The line search was not ready to suggest another configuration.')
                step_sizes.append(gpsls._step_size)
            else:
                num_val = original_num_val
                LOGGER.debug('None of the active line searches were ready to suggest a new configuration.')
                LOGGER.debug('We will initiate a new line search.')
                if cat_key in self._num_incumbents:
                    # We will set the default confiugration for this line search as
                    # the current incumbent
                    default_trial_id, default = self._num_incumbents[cat_key]
                    LOGGER.debug(f'We are starting a new GPS line search from the curent incumbent '
                                 f'for these numeric-valued parameters: {default}')
                    gpsls = GPSLS(self._config_space, num_hp, default, 
                        initial_step_size = np.mean(step_sizes) if len(step_sizes) > 0 else 1)
                    # The first suggestion is always the default
                    _, config_id = gpsls.suggest()
                    # Report the already-known loss values for the incumbent
                    for budget, loss in self._losses[default_trial_id].items():
                        gpsls.report(config_id, loss, budget)
                    # Track the fact that this configuration now reports to this new line
                    # search as well.
                    self._trial_id_to_gpsls[default_trial_id].append((gpsls, config_id))
                else:
                    LOGGER.debug(f'We are starting a new GPS line search from {num_val}')
                    # We will set the default configuration for this line search as
                    # the current values for num_val
                    gpsls = GPSLS(self._config_space, num_hp, num_val)
                num_val, config_id = gpsls.suggest()
                LOGGER.debug(f'The line search suggested {num_val}')
                self._gpsls[cat_key].append(gpsls)
            # Setup this trial id to report back to this line search
            self._trial_id_to_gpsls[trial_id].append((gpsls, config_id))
            config = self._to_configuration(num_hp, num_val, cat_hp, cat_val)
        # Convert to a Configuration for pretty printing
        configuration = Configuration(self._config_space, config)
        LOGGER.info(f'Suggesting the {configuration}')
        LOGGER.info(f'Which corresponds to the trial ID: {trial_id}')
        self._configurations[trial_id] = config
        return config

    @exception_logging()
    def on_trial_complete(self, trial_id, result, **kwargs):
        # The result can be None if an error occurred.
        if result is not None:
            self.on_trial_result(trial_id, result)

    @exception_logging()
    def on_trial_result(self, trial_id, result, update_model=True):
        LOGGER.debug('********** Recording a New Result **********')
        LOGGER.debug('Result:')
        LOGGER.debug(result)
        # Get the loss
        if self._metric not in result or not np.isfinite(result[self._metric]) or np.isnan(result[self._metric]):
            LOGGER.debug(f'Invalid metric in result: {result.get(self._metric, "[no result found]")}. '
                         f'Treating metric as {self._failed_result}')
            loss = self._failed_result
        else:
            loss = result[self._metric]*(1 if self._mode == 'min' else -1)
        # Get the budget spent evaluating this configuration
        budget = result[self._budget]
        LOGGER.info(f'Recorded a loss {loss} with fidelity {budget}')
        # Get the configuration and sort out the numeric and categorical hyperparameters
        config = self._configurations[trial_id]
        LOGGER.info(f'Corresponding configuration: {config}')
        num_hp, num_val, cat_hp, cat_val, cat_key = self._group_hps(config)

        if self._cat:
            LOGGER.debug('********** Reporting a result to the categorical BF-ASHA **********')
            ### Categorical BF-ASHA
            # Update the loss for this set of numeric values
            self._cat_rewards[cat_key][num_val] = loss
            # Find the best loss observed for this arm.
            best_loss = min(list(self._cat_rewards[cat_key].values()))
            cat_id, num_id = self._trial_id_to_cat_num_id[trial_id]
            # Report the value back to BF-ASHA
            self._cat_bfasha.report(cat_id, num_id, best_loss)
        # We don't need to  update this model if this categorical combination has no
        # numeric hyperparameters.
        if len(num_hp) > 0:
            ### Quadratic Model Approximation
            LOGGER.debug('********** GPS Line Search  **********')
            # Report the current result to all GPS line searchers that
            # contain this configuration
            for gpsls, config_id in self._trial_id_to_gpsls[trial_id]:
                LOGGER.debug(f'Reporting the result to {gpsls}')
                gpsls.report(config_id, loss, budget)
            # Update the loss record for this numeric-valued parameter configuration
            self._losses[trial_id][budget] = budget
            # Check to see if we need to update the incumbent
            if cat_key in self._num_incumbents:
                inc_id = self._num_incumbents[cat_key]
                if (max(list(self._losses[inc_id].keys()) + [-np.inf]) == budget 
                    and self._losses[inc_id][budget] > loss):
                    # Update the incumbent
                    LOGGER.debug(f'This configuration is replacing the incumbent that '
                                 f'had a loss of {self._losses[inc_id][fidelity]} because '
                                 f'its loss is {loss} for fidelity budget {budget}')
                    self._num_incumbents[cat_key] = (trial_id, num_val)
            else:
                LOGGER.debug(f'This is the first configuration to report, so it is the '
                             f'new incumbent with a loss of {loss} for fidelity budget {budget}')
                self._num_incumbents[cat_key] = (trial_id, num_val)

   
    def _group_hps(self, config):
        numeric = []
        categorical = []
        for hp in sorted(config.keys()):            
            if isinstance(self._config_space.get_hyperparameter(str(hp)), (Float, Integer)):
                numeric.append([hp, config[hp]])
            else:
                categorical.append([hp, config[hp]])
        numeric = np.array(numeric).reshape((-1, 2))
        categorical = np.array(categorical).reshape((-1, 2))
        LOGGER.debug(f'Numeric HPs:')
        LOGGER.debug(f'{numeric}')
        LOGGER.debug(f'Categorical HPs:')
        LOGGER.debug(f'{categorical}')
        num_hp = tuple(list(numeric[:,0]))
        num_val = tuple(list(numeric[:,1]))
        cat_hp = tuple(list(categorical[:,0]))
        cat_val = tuple(list(categorical[:,1]))
        cat_key = cat_hp + cat_val
        return num_hp, num_val, cat_hp, cat_val, cat_key

    def _to_configuration(self, num_hp, num_val, cat_hp, cat_val):
        def _numpy_to_python_dtype(x):
            return x.item() if isinstance(x, (np.ndarray, np.generic)) else x

        config = dict(zip([_numpy_to_python_dtype(cat) for cat in cat_hp],
                          [_numpy_to_python_dtype(val) for val in cat_val]))
        for hp, val in zip(num_hp, num_val):
            config[_numpy_to_python_dtype(hp)] \
                = int(val) if isinstance(self._config_space.get_hyperparameter(str(hp)), Integer) else float(val)
        return config


class GPSLS:
    """GPSLS

    Implements a single GPS line-search for a single set of numeric-valued
    parameters.
    """
    def __init__(self, config_space, num_hp, default, initial_step_size=1/GR, minimum_acceptable_fidelity=0, max_steps=5):
        # Fit a transformer for converting between normalized values and
        # actual parameter values
        self._transformer = Transformer()
        self._transformer.fit(num_hp, config_space)
        # Initialize the ID generate process
        self._next_id = 0
        self._id_prefix = generate_id() + '_{}'
        # Save the default configuration so we can suggest it later
        self._default = self._transformer.transform(default)
        # Initialize the bracket information
        self._values = [None, None]
        self._loss = [None, None]
        self._config_ids = [None, None]
        self._has_been_worse = [False, False]
        self._steps_taken = None
        # The initial step size should never be more than 1
        initial_step_size = min(initial_step_size, 1)
        self._step_size = initial_step_size
        self._initial_step_size = initial_step_size
        # The minimum fidelity budget that we consider sufficient
        # to use when deciding if a configuration is better than
        # another. 
        self._min_fidelity = minimum_acceptable_fidelity
        self._max_steps = max_steps
        self._n_suggestions = 0
       
    def report(self, config_id, loss, fidelity):
        LOGGER.debug('~~~~~~~~~~~~~~~ GPS LS report ~~~~~~~~~~~~~~~')
        LOGGER.debug(f'Registering a result for config={config_id}, loss={loss}, fidelity={fidelity}')
        # We might get reports for configurations that were
        # already kicked out of the bracket because a new
        # configuration was suggested. Ignore them.
        if config_id in self._config_ids:
            index = self._config_ids.index(config_id)
            self._loss[index][fidelity] = loss
            LOGGER.debug(f'Successfully recorded the result')
        else:
            LOGGER.debug(f'Ignoring the result because it was not for one of {self._config_ids}')

    def suggest(self):
        LOGGER.debug('~~~~~~~~~~~~~~~ GPS LS suggest ~~~~~~~~~~~~~~~')
        values = None
        config_id = None
        if self._values[0] is None and self._values[1] is None:
            # We haven't suggested anything yet! 
            # Suggest the default.
            LOGGER.debug('Nothing has been suggestedd yet, so we are '
                         'suggesting the default.')
            values, config_id = self._init_config(self._default, 0)
        elif self._values[1] is None:
            LOGGER.debug('We have only suggested a single point, so we are starting a new line search.')
            # Start a new line search
            values, config_id = self._take_random_step(1)
        else:
            LOGGER.debug('Checking to find the largest fidelity with which the configurations have '
                         'both been evaluated')
            # Find the large shared fidelity budget
            fidelity = max(list(set(list(self._loss[0].keys())).intersection(
                                set(list(self._loss[1].keys())))) + [-1])
            LOGGER.debug(f'The largest shared fidelity is {fidelity}')
            LOGGER.debug(f'The current reported losses are {self._loss}')
            if fidelity >= self._min_fidelity:
                # Both of the configurations
                # have a loss reported for at least one fidelity budget.
                # Eliminate the configuration with worse loss on it
                eliminated = 0 if self._loss[0][fidelity] > self._loss[1][fidelity] else 1
                LOGGER.debug(f'Eliminating index {eliminated} because its loss is '
                             f'{self._loss[eliminated][fidelity]} which is worse than '
                             f'{self._loss[1-eliminated][fidelity]}')
                # Check to see if we should start a new line search
                # or continue the old one
                LOGGER.debug(f'We have taken {self._steps_taken} steps along this line search')
                if self._steps_taken >= self._max_steps:
                    LOGGER.debug(f'Which is greater than {self._max_steps}, so we are starting a '
                                 f'new line search.')
                    # Start a new line search
                    values, config_id = self._take_random_step(eliminated)
                else:
                    LOGGER.debug(f'Taking the next step')
                    values, config_id = self._take_next_step(eliminated)
        LOGGER.debug(f'The resulting values we are suggesting are {values}')
        if values is not None:
            values = self._transformer.inverse_transform(values)
            LOGGER.debug(f'Which is {values} in the original space.')
            self._n_suggestions += 1
        return values, config_id

    def update_minimum_fidelity(self, minimum_acceptable_fidelity):
        self._min_fidelity = minimum_acceptable_fidelity

    def _take_random_step(self, eliminated):
        LOGGER.debug('Starting a new line search in a random direction.')
        # Take a step starting at the incumbent
        incumbent_values = self._values[1-eliminated]
        # Sample from the triangle distribution
        new_values = np.random.triangular(0, np.clip(incumbent_values, 0, 1), 1)
        # Adjust the step size to be the same as the previous step size in 
        # expectation
        step = new_values - incumbent_values
        self._steps_taken = 1
        new_values =  incumbent_values + step*self._step_size
        # Since we're taking a new random step, neither of them have been 
        # worse before.
        self._has_been_worse = [False, False]
        # If the most important parameter being optimized only has a small step
        # the overall step size can blow up. When starting a new line search
        # we cap it at its initial value divided by the square root of the number
        # of suggestions so as to not avoid getting stuck searching too far away
        # all the time and to ensure that we're slowly focusing on a smaller and
        # smaller portion of the landscape
        self._step_size = min(self._step_size, self._initial_step_size/np.sqrt(self._n_suggestions))
        # Initialize the new configuration
        return self._init_config(new_values, eliminated)

    def _take_next_step(self, eliminated):
        # Get the previous step
        step = self._values[1-eliminated] - self._values[eliminated]
        # If both sides have previously been worse, we were shrinking already
        was_shrinking = np.all(self._has_been_worse)
        LOGGER.debug(f'This bracket already was shrinking? {was_shrinking}')
        # If they are only just now worse, we are now going to shrink
        self._has_been_worse[self._get_side(step, eliminated)] = True
        is_shrinking = np.all(self._has_been_worse)
        LOGGER.debug(f'This bracket is now shrinking? {is_shrinking}')
        LOGGER.debug(f'Sides Eliminated: {self._has_been_worse}')
        # In the following, step_size is the current step_size relative to the
        # previous one and self._step_size is the current step size relative to
        # a new randomly sampled step.
        if not is_shrinking:
            # We are growing
            step_size = GR
            self._step_size *= GR
        elif not was_shrinking:
            # We were growing, but over-shot
            step_size = 1/GR - 1
            self._step_size /= 1 - 1/GR
        else:
            # We are shrinking
            step_size = 1/GR
            self._step_size /= GR
        # Calculate the new configuration from the step
        new_values = self._values[1-eliminated] + step*step_size
        # Increment the number of steps taken
        self._steps_taken += 1
        # Initialize the new configuration
        return self._init_config(new_values, eliminated)
        
    def _get_side(self, step, index):
        # We need to determine if the configuration idx is on the "left"
        # or the "right" side of the bracket, so that we know if we have
        # eliminated a configuration from both ends. We will arbitrarily
        # assign an ordering based on the ordering of the hyper-parameter
        # with the largest absolute value of the step size
        hp = np.argmax(np.abs(step))
        return 0 if self._values[index][hp] < self._values[1-index][hp] else 1

    def _init_config(self, values, index):
        self._values[index] = values
        self._loss[index] = {}
        self._config_ids[index] = self._get_id()
        return self._values[index], self._config_ids[index]

    def _get_id(self):
        next_id = self._id_prefix.format(self._next_id)
        self._next_id += 1
        return next_id


class Transformer:
    def __init__(self):
        self._lowers = None
        self._uppers = None
        self._lowers_original = None
        self._uppers_original = None
        self._integers = None
        self._logs = None

    def fit(self, numeric_hps, config_space):
        lowers = []
        uppers = []
        integers = []
        logs = []
        for hp in numeric_hps:
            hp = config_space.get_hyperparameter(str(hp))
            lowers.append(hp.lower)
            uppers.append(hp.upper)
            integers.append(isinstance(hp, Integer))
            logs.append(hp.log)
        self._integers = np.array(integers)
        self._logs = np.array(logs)
        self._lowers = np.array(lowers)
        self._uppers = np.array(uppers)
        self._lowers_original = copy.deepcopy(self._lowers)
        self._uppers_original = copy.deepcopy(self._uppers)
        self._lowers[self._logs] = np.log(self._lowers[self._logs])
        self._uppers[self._logs] = np.log(self._uppers[self._logs])

    def transform(self, X):
        X = copy.deepcopy(np.array(X, dtype=float))
        if X.ndim == 1:
            X[self._logs] = np.log(X[self._logs])
        elif X.ndim == 2:
            X[:,self._logs] = np.log(X[:,self._logs])
        else:
            raise ValueError(f'Wrong number of dimensions for X in transform(). '
                             f'Must be 1 or 2, but X has shape {X.shape}')
        X = X - self._lowers
        X = X/(self._uppers-self._lowers)
        return X

    def inverse_transform(self, X):
        X = copy.deepcopy(np.array(X))
        X = np.clip(X, 0, 1)
        X = X*(self._uppers-self._lowers)
        X = X + self._lowers
        if X.ndim == 1:
            X[self._logs] = np.exp(X[self._logs])
            X[self._integers] = np.round(X[self._integers])
        elif X.ndim == 2:
            X[:,self._logs] = np.exp(X[:,self._logs])
            X[:,self._integers] = np.round(X[:,self._integers])
        else:
            raise ValueError(f'Wrong number of dimensions for X in inverse_transform(). '
                             f'Must be 1 or 2, but X has shape {X.shape}')
        X = np.clip(X, self._lowers_original, self._uppers_original)
        return X

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

LOGGER = logging.getLogger('CQA')


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
    class CQASearcher(Searcher):
        """
        A ray.tune.Searcher wrapper around CoreCQASearcher
        
        See CoreCQASearcher.
        """
        def __init__(self, config_space, metric, max_t, *args, mode='min', **kwargs):
            LOGGER.debug('Initializing CQASearcher')
            super().__init__(metric=metric, mode=mode, **kwargs)
            LOGGER.debug('called super().__init__')
            self._cqa = CoreCQASearcher(config_space, metric, max_t, *args, mode=mode, **kwargs)
    
        def suggest(self, *args, **kwargs):
            self._cqa.suggest(*args, **kwargs)
    
        def on_trial_result(self, *args, **kwargs):
            self._cqa.on_trial_result(self, *args, **kwargs)
    
        def on_trial_complete(self, *args, **kwargs):
            self._cqa.on_trial_complete(self, *args, **kwargs) 


class CoreCQASearcher:
    """
    Parameters
    ----------
    config_space : ConfigSpace
      The configuration space.
    metric : str
      The name of the key in the dict of the results from the trainables to be
      optimized.
    max_t : float
      The maximum fidelity budget on which configurations are to be evaluated.
    grace_period : float
      The minimum fidelity budget on which configurations are to be evaluated.
    reduction_factor : int
      The reduction factor between milestones. This, grace_period and max_t
      are only used to determine when models should be fitted -- that is --
      any time a result is returned for a configuration that has just been
      evaluated on a milestone.
    mode : 'min' | 'max'
      Whether or not the metric is to be minimized or maximized.
    budget : str
      The name of the key in the dict of the results from the trainable that
      tracks the fidelity/budget upon which the configuration has been
      evaluated so far.
    failed_result : 1
      The value for metric that is to be assumed if the result is not reported,
      is reported as nan, or is reported as non-finite.
    random_fraction : float
      When a model is used to pick the values for numeric-valued
      hyperparameters, we will usually pick the optimizer of the model (unless
      it has already been evaluated). However, when run sequentially sometimes
      the model updates may be very small yielding nearly-identical optimizers.
      Hence, instead of evaluating the minimizer every time we instead sample
      from a Guassian centered around the optimizer of the model with this 
      probability. Must be a float in [0, 1].
    wall_time : str | None
      The name of the key in the dict of the results from the trainable that
      corresponds to the wallclock time cost of having evaluated the
      configuration. This should be non-cumulative over steps. Used to estimate
      how much time is left for the optimization process.
    wall_time_budget : float | None
      The maximum wall time allowed for configuration. Note that the searcher
      does not enforce this as a constraint in any way. Instead, it uses this
      as a means of estimating how much time is left for performing runs.
    model_fit_factor : float
      Models will only be retrained once they have at least this factor times
      the total fidelity budget used to evaluate the training data on which the
      last model was fit.
    max_concurrent_model_fits : int
      The models are fit in separate python processes. This limits the maximum
      number of models that can be fit at any one time. Must be at least 1.
    log_location : str | None
      Optionally enables logging to a file.
    log_level : int
      A log level, e.g., logging.DEBUG.
    log_formatter : logging.Formatter | str | None
      A log formatter or None to use the default.
    """
    @exception_logging()
    def __init__(self, config_space, metric, max_t, grace_period=1, reduction_factor=4, mode='min',
                 budget='budget', failed_result=1, random_fraction=0.5,
                 wall_time=None, wall_time_budget=None, model_fit_factor=1.5, 
                 max_concurrent_model_fits=1, window_size=0, log_location='CQA.log', log_level=logging.DEBUG,
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
        LOGGER.debug('********** Initializing the CQA Searcher **********')
        LOGGER.debug('**************************************************')
        self._config_space = config_space
        self._metric = metric
        self._mode = mode
        self._max_t = max_t
        self._grace_period = grace_period
        self._reduction_factor = reduction_factor
        MAX_RUNGS = int(np.log(max_t/grace_period)/np.log(reduction_factor) + 1)
        self._milestones = np.array([grace_period*reduction_factor**k
                                     for k in reversed(range(MAX_RUNGS))])
        self._failed_result = failed_result
        self._budget = budget
        self._wall_time = wall_time
        self._wall_time_budget = wall_time_budget
        self._start_time = time.time()
        self._random_fraction = random_fraction
        self._configurations = {}
        self._models = {}
        self._suggested_optimizer = defaultdict(bool)  # Initializes to False
        self._last_model_fidelity = defaultdict(lambda: None)
        process_manager = Manager()
        self._model_queue = process_manager.Queue()
        if max_concurrent_model_fits < 1:
            raise ValueError(f'max_concurrent_model_fits must be at least 1. '
                             f'Provided {max_concurrent_model_fits}.')
        self._max_processes = max_concurrent_model_fits
        self._processes_running = 0
        self._window_size = window_size
        self._model_fit_status = []
        self._model_fit_factor = model_fit_factor
        self._try_moonshots = False
        self._data = {}
        self._num_vals = {}
        self._cat_rewards = defaultdict(dict)
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
        self._y_transformers = defaultdict(RobustScaler)
        self._suggested_default = defaultdict(bool)  # initializes to False
        self._suggested_default_cat = False
        LOGGER.debug('**************************************************')
        LOGGER.debug('********** Initialized the CQA Searcher **********')
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
            self._check_for_models()
            # Now check to see if this combination of categorical values has a model for the
            # numeric hyper-parameters (if any exist)
            if cat_key in self._models:
                LOGGER.debug('Found a model for the numeric hyperparameters of the configuration.')
                # Get the model for these categorical_values.
                model = self._models[cat_key]
                # And get a configuration for the numerical parameters predicted to be of high quality
                num_val = self._get_next_config(
                    model, self._suggested_optimizer[cat_key], 
                    self._data[cat_key], self._num_vals[cat_key],
                    self._transformers[cat_key], self._random_fraction,
                    self._last_model_fidelity[cat_key])
                # If it wasn't already suggested, then we're about to.
                self._suggested_optimizer[cat_key] = True
            else:
                if False and not self._suggested_default[cat_key]:
                    LOGGER.debug('We are starting the search amongst this set of numeric-valued '
                                 'hyper-parameters by recommending the defaults.')
                    num_val = [self._config_space.get_hyperparameter(str(hp)).default_value for hp in num_hp]
                    self._suggested_default[cat_key] = True
                else:
                    # Transform the numeric hyper-parameters to a unit cube
                    if cat_key not in self._transformers:
                        transformer = Transformer()
                        transformer.fit(num_hp, self._config_space)
                        self._transformers[cat_key] = transformer
                    num_val = np.random.random(len(num_val))
                    LOGGER.debug('Sampling a new set of numeric values at random.')
                    num_val = self._transformers[cat_key].inverse_transform(num_val)
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
            LOGGER.debug('********** Quadratic Model Approximation **********')
            # Accumulate the numeric data for this set of categorical values 
            if cat_key not in self._data:
                self._data[cat_key] = []
                self._num_vals[cat_key] = {}
            self._data[cat_key].append((trial_id, budget, loss))
            self._num_vals[cat_key][trial_id] = num_val

            if update_model:
                # Handle any back-log of fitted models so that we know if there
                # are any free processes available.
                self._check_for_models()
                if self._processes_running < self._max_processes:
                    LOGGER.debug(f'The number of processes running {self._processes_running} is less than '
                                 f'the maximum allowed {self._max_processes}, so we are going to initiate '
                                 f'a new model-fitting process.')
                    # Gets the data with the best fidelity available for training,
                    # provied that we have collected enough new data since the last
                    # model fit. 
                    X_train, y_train, budgets, current_fidelity = self._get_data_for_training(
                        self._data[cat_key], self._num_vals[cat_key], len(num_val), self._last_model_fidelity[cat_key])
                    min_n_samples_separable = SCQA.min_samples(len(num_val))
                    if X_train is not None:
                        X_train = self._transformers[cat_key].transform(X_train)
                        LOGGER.debug('Transformed X_train:')
                        LOGGER.debug(X_train)
                        # Scale the losses so that the constraints on the parameters
                        # in the model don't make the optimal solution infeasable
                        y_transformer = RobustScaler()
                        y_train_untransformed = y_train
                        y_train = y_transformer.fit_transform(y_train.reshape(-1, 1)).squeeze()
                        LOGGER.debug('Transformed y_train:')
                        LOGGER.debug(y_train)
                        LOGGER.debug('Sample_weights (fidelity budgets):')
                        LOGGER.debug(budgets)
                        p = Process(target=fit_model,
                                    args=(X_train, y_train, budgets, self._model_queue, 
                                          cat_key, y_transformer, current_fidelity))
                        p.start()
                        self._processes_running += 1
                        LOGGER.debug(f'We now have {self._processes_running} model processes running.')
                else:
                    LOGGER.debug(f'There were too many processes running {self._processes_running} to '
                                 f'start another one because the maximum allowed processes is '
                                 f'{self._max_processes}')
            else:
                LOGGER.debug('Skipping model-fitting because update_model=False')
    
    def _check_for_models(self):
        while self._model_queue.qsize() > 0:
            success_, model_type, model_dict, X_train, y_train, budgets, args, logs, exception = self._model_queue.get()
            self._processes_running -= 1
            cat_key, y_transformer, current_fidelity = args
            LOGGER.debug('Found a new result in the model queue.')
            LOGGER.debug(f'We now have {self._processes_running} model processes running.')
            LOGGER.debug('The following log messages came from the model-fitting process:')
            for log in logs:
                LOGGER.debug(log)
            if exception is not None:
                LOGGER.debug('The worker process encountered the following exception.', exc_info=exception)
            if success_:
                LOGGER.debug(f'It was a successful model fit for {cat_key}')
                model = GCQA() if model_type == 'GCQA' else SCQA()
                model.from_dict(model_dict)
                y_train = y_transformer.inverse_transform(y_train.reshape((-1, 1))).squeeze()
                if cat_key in self._models:
                    old_pred = self._y_transformers[cat_key].inverse_transform(
                        self._models[cat_key].predict(X_train).reshape((-1, 1))).squeeze()
                    old_score = np.mean(np.abs(old_pred - y_train)*budgets)/np.sum(budgets)
                else:
                    old_score = None
                new_pred = y_transformer.inverse_transform(model.predict(X_train).reshape((-1, 1))).squeeze()
                new_score = np.mean(np.abs(new_pred - y_train)*budgets)/np.sum(budgets)
                LOGGER.debug(f'The MAE of the new model is {new_score}')
                LOGGER.debug(f'The MAE of the old model is {old_score}')
                if old_score is None or new_score <= old_score:
                    self._models[cat_key] = model
                    self._y_transformers[cat_key] = y_transformer
                    self._suggested_optimizer[cat_key] = False
                    self._last_model_fidelity[cat_key] = current_fidelity
                    LOGGER.debug('Accepted the new model.')
                    self._model_fit_status.append('Success')
                else:
                    LOGGER.debug('Rejected the new model; keeping the old one for now.')
                    self._model_fit_status.append('Regression')
            else:
                LOGGER.debug(f'It was an unsuccessful model fit for {cat_key}')
                self._model_fit_status.append('Unsuccessful')
            LOGGER.debug(f'History of model-fitting process result statuses:\n'
                         f'{pd.Series(self._model_fit_status).value_counts()}')

    def _get_data_in_window(self, df, fidelity, window_size, rtol=1e-2, atol=1e-2):
        # Different backends calculate budget fidelities in slightly different ways
        # causing floating point errors. Put in a generous amount of tolerance
        # into the fidelity window bounds.
        fidelity_lower = fidelity - (atol + rtol*np.abs(fidelity))
        fidelity_upper = fidelity*self._reduction_factor**window_size
        fidelity_upper = fidelity_upper + (atol + rtol*np.abs(fidelity))
        window = np.logical_and(fidelity_lower <= df['Fidelity'], 
                                df['Fidelity'] <= fidelity_upper)
        # Grab only the result with the largest fidelity for each configuration.
        data_in_window = df[window].groupby('ID').tail(1)
        return data_in_window

    def _get_data_for_training(self, data, num_vals, n_dim, fidelity):
        # Setup
        min_n_samples_separable = SCQA.min_samples(n_dim)
        last_data = None
        df = _data_to_df(data)
        if fidelity is None and len(df) > 0:
            fidelity = df['Fidelity'].min()
        data_in_window = self._get_data_in_window(df, fidelity, self._window_size)
        LOGGER.debug(f'Checking to find the largest fidelity budget window with sufficient data '
                     f'to fit a model. '
                     f'Minimum number of training examples to fit model: {min_n_samples_separable}')
        LOGGER.debug(f'Data that is avialable:\n{df}')
        LOGGER.debug(f'Budget fidelity window {fidelity} to '
                     f'{fidelity*self._reduction_factor**self._window_size} has '
                     f'{len(data_in_window)} data samples available.')
        # Make the window as large as possible while there is still enough data.
        while len(data_in_window) >= min_n_samples_separable:
            last_data = data_in_window
            fidelity *= self._reduction_factor
            data_in_window = self._get_data_in_window(df, fidelity, self._window_size)
            LOGGER.debug(f'Budget fidelity window {fidelity} to '
                         f'{fidelity*self._reduction_factor**self._window_size} has '
                         f'{len(data_in_window)} data samples available.')

        # We only fit a new model if we found a window with enough data        
        if last_data is not None:
            X_train = np.array(list(last_data['ID'].map(num_vals)), dtype=float)
            y_train = np.array(last_data['Loss'], dtype=float)
            budgets = np.array(last_data['Fidelity'], dtype=float)
            fidelity /= self._reduction_factor
        else:
            X_train = None
            y_train = None
            budgets = None
        return X_train, y_train, budgets, fidelity
           
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

    def _get_next_config(self, model, suggested_optimizer, data, num_vals, transformer, random_fraction,
                         last_model_fidelity,  n_samples=3, top_k=3):
        df = _data_to_df(data)
        n_configs_evaluated = len(df.groupby('ID').tail(1))
        df = self._get_data_in_window(df, last_model_fidelity, window_size=2)
        # Keep only the largest fidelity for each configuration
        X_train = np.array(list(df['ID'].map(num_vals)), dtype=float)
        y_train = np.array(df['Loss'], dtype=float) 
        x_inc = X_train[np.argmin(y_train)]
        LOGGER.debug(f'The current incumbent is: {x_inc}')
        x_inc = transformer.transform(x_inc)
        LOGGER.debug(f'Or, when transformed: {x_inc}')
        x_next = None
        # If we haven't yet suggested it, suggest the optimizer of the model
        if self._try_moonshots and not suggested_optimizer: 
            x_star = model.get_minimizer()
            LOGGER.debug(f"The model's minimizer (when encoded) is: {x_star}")
            # Check if the optimizer if feasable
            if not (np.all(x_star >= 0) and np.all(x_star <= 1)):
                LOGGER.debug(f'Since the minimizer is not feasable we are going to progressively '
                             f'do line searches along the eigenvectors with the smallest eigenvalues '
                             f'to make the minimizer more like the incumbent, until it is feasable.')
                _, _, H = model.get_model()
                eigenvalues, eigenvectors = np.linalg.eig(H)
                order = np.argsort(eigenvalues)
                idx = 0
                while not (np.all(x_star >= 0) and np.all(x_star <= 1)) and idx < len(order):
                    # Do a line search along the eigenvalue with the minimum, unused eigenvalue that
                    # seeks to minimize the distance between the optimizer and the current incumbent.
                    res = minimize_scalar(
                        lambda alpha: np.linalg.norm(x_inc - (x_star + alpha*eigenvectors[order[idx]])))
                    # Move to the solution closest to the current incumbent
                    x_star += eigenvectors[order[idx]]*res.x
                    # Update to the next eigenvalue in case we're still not feasable.
                    idx += 1
            if not (np.all(x_star >= 0) and np.all(x_star <= 1)):
                LOGGER.debug(f'Something went wrong. It is still infeasable: {x_star}')
            elif np.allclose(x_star, x_inc, rtol=0.01):
                LOGGER.debug(f'The minimizer ended up being nearly identical to the incumbent, so we will '
                             f'use the sampling method instead.')
            else:
                x_next = x_star
                LOGGER.debug(f'Now that it is feasable the minimizer is: {x_star}')
        # Otherwise, sample from around the incumbent a handful of times and use the model to
        # pick the one expected to be the best.
        if x_next is None:
            LOGGER.debug(f'Sampling from a Guassian around the incumbent')
            # Pick the variance of the Guassian so that it slowly shrinks as we get more data,
            # focusing us ever-more-tightly around the incumbent.
            #var = (0.25/n_configs_evaluated**0.5 + 0.01)**2
            # One standard deviation from the mean covers 10% of each parameter's range.
            var = (0.05)**2
            # TODO: Cleanup and expose this as a parameter
            LOGGER.debug(f'Using a variance of {var}')
            # Sample n_samples configurations and reject any that are infeasable.
            LOGGER.debug(f'Sampling {n_samples} configurations for each of the top {top_k} configurations.')
            X_next = []
            for idx in np.argpartition(y_train, top_k)[:top_k]:
                x_top = transformer.transform(X_train[idx])
                x_next = np.random.multivariate_normal(x_top, cov=np.diag([var]*len(x_inc)), size=n_samples)
                X_next.append(x_next)
            x_next = np.concatenate(X_next, axis=0)
            feasable = np.logical_and(np.all(x_next >= 0, axis=1), np.all(x_next <= 1, axis=1))
            if np.any(feasable):
                # Grab the first feasable configurationlgorithm.
                x_next = x_next[feasable]
                LOGGER.debug(f'Found {len(x_next)} feasable configurations for the numeric hyperparameters.')
                x_next = x_next[np.argmin(model.predict(x_next))]
                LOGGER.debug('Picked the one with the best score predicted by the mdoel.')
            else:
                # Sample a random configuration from the unit cube to
                # help improve the model.
                x_next = np.random.random(len(x_inc))
                LOGGER.debug('None of the sampled configurations were feasable. Sampling '
                             'uniformly at random from the unit cube instead.')
        x_next = transformer.inverse_transform(x_next)
        LOGGER.debug(f'The new set of numeric values to evaluate (in the original space) are {x_next}')
        return x_next


def _data_to_df(data):
    df = pd.DataFrame(data, columns=['ID', 'Fidelity', 'Loss'])
    df = df.sort_values('Fidelity')
    return df


def fit_model(X_train, y_train, budgets, queue, *extra_args):
    # multi-processing and logging in python can cause deadlocks. There are
    # workarounds, but rather than spending effort figuring out those, we're
    # just going to make a list of logging statements, send them back with
    # the result and log them when we collect the result.
    logs = []
    exception = None
    logs.append('A model fitting process is starting')
    n_samples = len(X_train)
    min_n_samples_general = GCQA.min_samples(X_train.shape[1])
    if n_samples >= min_n_samples_general:
        logs.append('Fitting a general convex quadratic under-estimator model.')
        model = GCQA()
        model_type = 'GCQA'
    else:
        logs.append(f'Number of training examples is less than the minimum number '
                    f'({min_n_samples_general}) required to fit a general convex '
                    f'quadratic model, so we are fitting a separable one instead.')
        model = SCQA()
        model_type = 'SCQA'
    success_ = False
    try:
        model.fit(X_train, y_train, sample_weight=budgets)
        model_dict = model.to_dict()
        success_ = True
        logs.append('Successfully fit the model.')
    except Exception as e:
        # Something went wrong. Don't update the model.
        logs.append('Something went wrong with the model fitting and scoring process, '
                    'falling back on the previous model, if available.')
        exception = e
        model_dict = None
    logs.append('A model fitting process is ending')
    logs = []
    exception = None
    queue.put((success_, model_type, model_dict, X_train, y_train, budgets, extra_args, logs, exception))


class Transformer():
    def __init__(self):
        self._lowers = None
        self._uppers = None
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
        self._lowers[self._logs] = np.log(self._lowers[self._logs])
        self._uppers[self._logs] = np.log(self._uppers[self._logs])

    def transform(self, X):
        X = copy.deepcopy(np.array(X))
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
        return X

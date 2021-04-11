import time
import copy 
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import RobustScaler

from ray.tune.suggest import Searcher

from ConfigSpace.hyperparameters import UniformFloatHyperparameter as Float
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as Integer
from ConfigSpace import Configuration

from .cqa import SCQA, GCQA

LOGGER = logging.getLogger('CQA')

class CQASearcher(Searcher):
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
    categorical_grace_period : int
      The minimum number of results that must have been reported -- summed over
      all configurations with a given set of categorical values --  before that
      set of categorical values may be eliminated from consideration.
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
    lambda_ : float
      Must be non-negative. Weight of the L1-regularizer used when fitting
      the general convex quadratic models. (Not used for the separable
      convex quadratic models.)
    norm : float
      The norm used in the objective function of the general convex quadratic
      models. Note that the separable convex quadratic model always optimizes
      the L1-norm regardless.
    log_location : str | None
      Optionally enables logging to a file.
    log_level : int
      A log level, e.g., logging.DEBUG.
    log_formatter : str | None
      A log formatter or None to use the default.
    """
    def __init__(self, config_space, metric, max_t, grace_period=1, reduction_factor=4, mode='min',
                 budget='budget', categorical_grace_period=20, failed_result=1, random_fraction=0.5,
                 wall_time=None, wall_time_budget=None, model_fit_factor=1.5, lambda_=0, norm=1, 
                 log_location='CQA.log', log_level=logging.DEBUG, log_formatter=None, **kwargs):
        super().__init__(metric=metric, mode=mode, **kwargs)
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
        self._model_weights = defaultdict(int)  # Initializes to 0
        self._model_fit_factor = model_fit_factor
        self._lambda = lambda_
        self._norm = norm
        self._data = {}
        self._cat_rewards = defaultdict(list)
        self._cat_times = defaultdict(list)
        self._cat_eliminated = []
        self._cat_candidates = set([])
        self._categorical_grace_period = categorical_grace_period
        self._transformers = {}
        self._y_transformers = defaultdict(RobustScaler)
        if log_location is not None:
            fh = logging.FileHandler(log_location)
            fh.setLevel(log_level)
            if log_formatter is not None:
                fh.setFormatter(log_formatter)
            LOGGER.addHandler(fh)
        LOGGER.setLevel(log_level)
        LOGGER.debug('**************************************************')
        LOGGER.debug('********** Initialized the CQA Searcher **********')
        LOGGER.debug('**************************************************')

    def suggest(self, trial_id):
        eliminated = True
        LOGGER.debug('********** Suggesting a Configuration **********')
        while eliminated:
            config = self._config_space.sample_configuration()
            LOGGER.debug(f'Sampled a new {config}')
            # Convert to a dict to work with internally
            config = dict(config)
            # Check to see if this combination of categorical values has been eliminated
            num_hp, num_val, cat_hp, cat_val, cat_key = self._group_hps(config)
            eliminated = cat_key in self._cat_eliminated
            LOGGER.debug(f'This categorical combination has been eliminated? {eliminated}')
        # Now check to see if this combination of categorical values has a model for the
        # numeric hyper-parameters (if any exist)
        if cat_key in self._models and len(num_hp) > 0:
            LOGGER.debug('Found a model for the numeric hyperparameters of the configuration.')
            # Get the model for these categorical_values.
            model = self._models[cat_key]
            # And get a configuration for the numerical parameters predicted to be of high quality
            num_val = _get_next_config(
                model, self._suggested_optimizer[cat_key], 
                self._data[cat_key], self._transformers[cat_key], self._random_fraction)
            # If it wasn't already suggested, then we're about to.
            self._suggested_optimizer[cat_key] = True
            config = self._to_configuration(num_hp, num_val, cat_hp, cat_val)
        # Convert to a Configuration for pretty printing
        configuration = Configuration(self._config_space, config)
        LOGGER.info(f'Suggesting the {configuration}')
        LOGGER.info(f'Which corresponds to the trial ID: {trial_id}')
        self._configurations[trial_id] = config
        return config

    def on_trial_complete(self, trial_id, result, **kwargs):
        # The result can be None if an error occurred.
        if result is not None:
            self.on_trial_result(trial_id, result)

    def on_trial_result(self, trial_id, result):
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

        # If this set of categorical values has already been eliminated, we 
        # don't need to update the cat rewards for it
        if cat_key not in self._cat_eliminated and len(cat_hp) > 0:
            LOGGER.debug('********** Starting Rising Bandits **********')
            ### Rising Bandits
            # Update the reward and time for this combination of categorical values
            if len(self._cat_rewards[cat_key]) > 0:
                self._cat_rewards[cat_key].append(min(loss, self._cat_rewards[cat_key][-1]))
            else:
                self._cat_rewards[cat_key].append(loss)
            if self._wall_time is not None and self._wall_time in result:
                self._cat_times[cat_key].append(result[self._wall_time])
    
            # Create a helper for extracting cat speed and other data
            def _extract(cat_rewards, cat_times):
                current_pull = len(cat_rewards)
                previous_pull = int(current_pull/2)
                current_reward = cat_rewards[-1]
                previous_reward = cat_rewards[-1]
                # Linearize over the most recent half of the reward data
                speed = (current_reward - previous_reward)/(current_pull - previous_pull)
                # Get the mean time spent
                time_cost = np.mean(cat_times) if len(cat_times) > 0 else None
                return speed, current_reward, current_pull, time_cost
    
            # Get the number of times we have pulled this categorical arm
            if len(self._cat_rewards[cat_key]) >= self._categorical_grace_period:
                LOGGER.debug('Current cat rewards:')
                LOGGER.debug(self._cat_rewards)
                # Get the current reward, the speed of improvement, and the number of pulls
                cat_speed, cat_reward, cat_n_pulls, cat_mean_time = _extract(
                    self._cat_rewards[cat_key], self._cat_times[cat_key])
                LOGGER.debug(f'Current arm reward {cat_reward} after {cat_n_pulls} pulls '
                             f'with speed {cat_speed} per {cat_mean_time}')
                # Compare the expected future reward to each other candidate
                for other_key in self._cat_rewards:
                    if (len(self._cat_rewards[other_key]) > self._categorical_grace_period 
                        and other_key not in self._cat_eliminated):
                        # Do the same linearization of the reward
                        other_speed, other_reward, other_n_pulls, other_mean_time = _extract(
                            self._cat_rewards[other_key], self._cat_times[other_key])
                        LOGGER.debug(f'Other arm reward {other_reward} after {other_n_pulls} pulls '
                                     f'with speed {other_speed} per {other_mean_time}')
                        if (cat_mean_time is not None
                            and other_mean_time is not None
                            and self._wall_time_budget is not None):
                            # Calculate the estimated upper bound for the amount of time remaining.
                            time_remaining = self._wall_time_budget - (time.time() - self._start_time)
                            LOGGER.debug(f'Approximately {time_remaining:.2f} seconds remaining')
                            cat_lower = cat_reward + cat_speed*time_remaining/cat_mean_time
                            other_lower = other_reward + other_speed*time_remaining/other_mean_time
                        else:
                            # Calculate the upper bound on each at 2*max_n_pulls steps
                            future_n_pulls = 2*max(cat_n_pulls, other_n_pulls)
                            LOGGER.debug(f'Assuming {future_n_pulls/2} pulls as a total budget')
                            cat_lower = cat_reward + cat_speed*(future_n_pulls - cat_n_pulls)
                            other_lower = other_reward + other_speed*(future_n_pulls - other_n_pulls)
                        LOGGER.debug(f'Estimated bound on future reward for current arm: {cat_lower}')
                        LOGGER.debug(f'Estimated bound on future reward for other arm:   {other_lower}')
                        if cat_reward < other_lower:
                            # Eliminate other
                            self._cat_eliminated.append(other_key)
                            LOGGER.debug('Eliminated the other arm')
                        elif other_reward < cat_lower:
                            # Eliminate cat
                            self._cat_eliminated.append(cat_key)
                            LOGGER.debug('Eliminated the current arm')
                            # It's unlikely this will eliminate anything else.
                            break
        # If this set of categorical values has now been elminated, we also
        # don't need to update the model for its numeric data. We also don't
        # need tup update this model if this categorical combination has no
        # numeric hyperparameters.
        if cat_key not in self._cat_eliminated and len(num_hp) > 0:
            ### Quadratic Model Approximation
            LOGGER.debug('********** Quadratic Model Approximation **********')
            # Accumulate the numeric data for this set of categorical values 
            if cat_key not in self._data:
                self._data[cat_key] = {}
            self._data[cat_key][num_val] = (budget, loss)
            # We only fit a new model if there is enough data to do so
            n_samples = len(self._data[cat_key])
            min_n_samples_separable = SCQA.min_samples(len(num_val))
            LOGGER.debug(f'Number of training examples: {n_samples}; '
                         f'Minimum number of training examples to fit model: {min_n_samples_separable}')
            if n_samples >= min_n_samples_separable:
                min_n_samples_general = GCQA.min_samples(len(num_val))
                if n_samples >= min_n_samples_general:
                    LOGGER.debug('Fitting a general convex quadratic under-estimator model.')
                    model = GCQA(lambda_=self._lambda, norm=self._norm)
                else:
                    LOGGER.debug(f'Number of training examples is less than the minimum number '
                                 f'({min_n_samples_general}) required to fit a general convex '
                                 f'quadratic model, so we are fitting a separable one instead.')
                    model = SCQA()
                X_train = list(self._data[cat_key].keys())
                data = np.array([self._data[cat_key][x] for x in X_train])
                X_train = np.array(X_train, dtype=float)
                y_train = data[:,1]
                budgets = data[:,0]
                # Only fit a new model if we have observed a sufficient amount of new data
                # to warrant the cost of the model fitting process.
                if np.sum(budgets) >= self._model_weights[cat_key]*self._model_fit_factor:
                    # Transform the numeric hyper-parameters to a unit cube
                    if cat_key not in self._transformers:
                        transformer = Transformer()
                        transformer.fit(num_hp, self._config_space)
                        self._transformers[cat_key] = transformer
                    X_train = self._transformers[cat_key].transform(X_train)
                    LOGGER.debug('Transformed X_train:')
                    LOGGER.debug(X_train)
                    # Scale the losses so that the constraints on the parameters
                    # in the model don't make the optimal solution infeasable
                    y_transformer = RobustScaler()
                    y_train = y_transformer.fit_transform(y_train.reshape(-1, 1)).squeeze()
                    LOGGER.debug('Transformed y_train:')
                    LOGGER.debug(y_train)
                    LOGGER.debug('Sample_weights (fidelity budgets):')
                    LOGGER.debug(budgets)
                    try:
                        model.fit(X_train, y_train, sample_weight=budgets)
                        LOGGER.debug('Successfully fit the model.')
                        # Make sure the new model hasn't diverged to something worse
                        # than the old one
                        if cat_key in self._models:
                            old_pred = self._y_transformers[cat_key].inverse_transform(
                                self._models[cat_key].predict(X_train).reshape((-1, 1))).squeeze()
                            old_score = np.mean(np.abs(old_pred - data[:,1])*budgets)/np.sum(budgets)
                        else:
                            old_score = None
                        new_pred = y_transformer.inverse_transform(model.predict(X_train).reshape((-1, 1))).squeeze()
                        new_score = np.mean(np.abs(new_pred - data[:,1])*budgets)/np.sum(budgets)
                        LOGGER.debug(f'The MAE of the new model is {new_score}')
                        LOGGER.debug(f'The MAE of the old model is {old_score}')
                        if old_score is None or new_score <= old_score:
                            self._models[cat_key] = model
                            self._y_transformers[cat_key] = y_transformer
                            self._suggested_optimizer[cat_key] = False
                            LOGGER.debug('Accepted the new model.')
                        else:
                            LOGGER.debug('Rejected the new model; keeping the old one for now.')
                    except Exception as e:
                        # Something went wrong. Don't update the model.
                        LOGGER.debug('Something went wrong with the model fitting and scoring process, '
                                     'falling back on the previous model, if available.', exc_info=e)
                    # Update the model weight even if something failed, so we don't get stuck wasting
                    # Time constantly trying to re-build this model when the data has hardly changed/
                    self._model_weights[cat_key] = np.sum(budgets)
                else:
                    LOGGER.debug(f'Skipping model fitting because the available training '
                                 f'data has used a combined fidelity budget of '
                                 f'{np.sum(budgets)}, which is less than the required amount: '
                                 f'{self._model_fit_factor}*{self._model_weights[cat_key]}')
           
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


def _get_next_config(model, suggested_optimizer, data, transformer, random_fraction, n_samples=5):
    X_train = list(data.keys())
    data = np.array([data[x] for x in X_train])
    X_train = np.array(X_train, dtype=float)
    y_train = data[:,1]
    x_inc = X_train[np.argmin(y_train)]
    LOGGER.debug(f'The current incumbent is: {x_inc}')
    x_inc = transformer.transform(x_inc)
    LOGGER.debug(f'Or, when transformed: {x_inc}')
    x_next = None
    # If we haven't yet suggested it, suggest the optimizer of the model
    if not suggested_optimizer: 
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
        var = (2/len(X_train) + 0.05)**2
        LOGGER.debug(f'Using a variance of {var}')
        # Sample n_samples configurations and reject any that are infeasable.
        LOGGER.debug(f'Sampling {n_samples} configurations.')
        x_next = np.random.multivariate_normal(x_inc, cov=np.diag([var]*len(x_inc)), size=n_samples)
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


old = '''
def _get_next_config(model, suggested, data, transformer, random_fraction, n_samples=50):
    train_size = len(data)
    x_star = model.get_minimizer()
    LOGGER.debug(f"The model's minimizer (when encoded) is: {x_star}")
    # We use the transformer here because it also clips the configuration
    # to make it feasible if need be since the model's optimizer may not be
    # feasible.
    x_star = np.clip(x_star, 0, 1)
    LOGGER.debug(f'When clipped to be feasable it is: {x_star}')
    # If a configuration very similar to this one been evaluated before,
    # sample a new one instead. Also sample a random fraction of the
    # configurations anyways to help ensure some diversity. The random
    # fraction helps inject diversity when only non-important hyper-
    # parameters are changing between configurations.
    similar = len(suggested) > 0 and np.isclose(x_star, suggested, rtol=0.01).all(axis=1).any()
    force_random = np.random.random() < random_fraction 
    if similar or force_random:
        LOGGER.debug(f'Sampling from a Guassian instead of returning the minimizer because:\n'
                     f'  - Something similar has been evaluated before? {similar}.\n'
                     f'  - This is one of {random_fraction*100:.2f}% forced-random samples? {force_random}.')
        # Get the model's hessian.
        _, _, H = model.get_model()
        LOGGER.debug("The model's Hessian is")
        LOGGER.debug(H)
        # And get a covariance matrix of a multi-variate Guassian that
        # we can sample from using the Hessian.
        cov = _get_covariance(H, train_size)
        LOGGER.debug("The Guassian's covariance matrix is")
        LOGGER.debug(cov)
        # Sample n_samples configurations and reject any that are infeasable.
        LOGGER.debug(f'Sampling {n_samples} configurations to check for a feasable one.')
        x_next = np.random.multivariate_normal(x_star, cov=cov, size=n_samples)
        feasable = np.logical_and(np.all(x_next >= 0, axis=1), np.all(x_next <= 1, axis=1))
        if np.any(feasable):
            # Grab the first feasable configuration.
            x_next = x_next[np.where(feasable)[0][0]]
            LOGGER.debug('Found a feasable configuration for the numeric hyperparameters')
        else:
            # Sample a random configuration from the unit cube to
            # help improve the model.
            x_next = np.random.random(len(x_star))
            LOGGER.debug('None of the sampled configurations were feasable. Sampling '
                         'uniformly at random from the unit cube instead.')
    else:
        x_next = x_star
    suggested.append(x_next)
    x_next = transformer.inverse_transform(x_next)
    LOGGER.debug(f'The new set of numeric values to evaluate (in the original space) are {x_next}')
    return x_next


def _get_covariance(H, n):
    """
    Parameters
    ----------
    H : np.ndarray
      The Hessian of the fitted quadratic model.
    n : int
      The number of training examples used to fit the model.

    Returns
    -------
    np.ndarray
       The covariance matrix of a multi-variate Guassian distribution
       that can be used to sample configurations expected to be of a
       high quality if centered around the optimum of the model.
    """
    # Get the inverse Hessian so that eigenvectors with large eigenvalues
    # have eigenvalues that are small instead. (This way we sample less far
    # away along vectors along with the model grows quickly.)
    H_inv = np.linalg.inv(H)
    # Get the eigenvalues of the inverse Hessian
    eigenvalues, _ = np.linalg.eigh(H_inv)
    LOGGER.debug(f'The eigenvalues of the inverse Hessian are {eigenvalues}')
    # For simplicitly, assume we have a uni-variate quadratic scaled by the
    # maximum eigenvalue of H (which is one over the minimum eigenvalue of
    # H_inv) that is centered at 0.5 and has a domain of [0, 1], and we want
    # to obtain the length x from 0.5 such that all f(x) <= y*f(0.5), where y
    # is in (0, 1) and progressively shrinks as the search process progresses.
    # If y=0.1, then this means we want to sample from a Guassian such that all
    # samples within the first standard deviation from the mean are within 10%
    # of optimal in the objective space.
    def f_inv(x):
        # Multiplying by the minimum eigenvalue of H_inv is equivalent to
        # dividing by the maximum eigenvalue of H.
        return np.sqrt(x*np.min(eigenvalues))
    # Set y to cut in half every time the number of samples doubles and square
    # to convert standard deviation to variance.
    percent_from_optimal = 0.5**(np.log(n*2/4)/np.log(2))
    LOGGER.debug(f'Sampling with one standard deviation equal to approximately '
                 f'{percent_from_optimal*100:.3f}% from the optimal objective '
                 f'function value.')
    min_var = f_inv(percent_from_optimal)**2
    LOGGER.debug(f'Adjusting the inverse Hessian so that the eigenvector for '
                 f'the minimum eigenvalue corresponds to a standard devaition '
                 f'of {np.sqrt(min_var):.5f}')
    # Rescale the inverse Hessian
    return H_inv*min_var/np.min(eigenvalues)
'''


class Transformer():
    def __init__(self):
        self._lowers = None
        self._uppers = None
        self._integers = None

    def fit(self, numeric_hps, config_space):
        lowers = []
        uppers = []
        integers = []
        for hp in numeric_hps:
            hp = config_space.get_hyperparameter(str(hp))
            lowers.append(hp.lower)
            uppers.append(hp.upper)
            integers.append(isinstance(hp, Integer))
        self._lowers = np.array(lowers)
        self._uppers = np.array(uppers)
        self._integers = np.array(integers)

    def transform(self, X):
        X = copy.deepcopy(np.array(X))
        X = X - self._lowers
        X = X/(self._uppers-self._lowers)
        return X

    def inverse_transform(self, X):
        X = copy.deepcopy(np.array(X))
        X = np.clip(X, 0, 1)
        X = X*(self._uppers-self._lowers)
        X = X + self._lowers
        if X.ndim == 1:
            X[self._integers] = np.round(X[self._integers])
        elif X.ndim == 2:
            X[:, self._integers] = np.round(X[:, self._integers])
        return X

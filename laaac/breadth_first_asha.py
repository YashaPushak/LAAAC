from collections import defaultdict
from typing import Optional
import logging

import numpy as np

from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler as ASHA
from ray.tune.schedulers.async_hyperband import _Bracket
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.trial import Trial
from ray.tune import trial_runner



class BreadthFirstASHA(ASHA):
    """BreadthFirstASHA

    Implements the breadth-first asynchronous successive halving algorithm.
    This should provide similar theoretical performance to the regular ASHA
    but with better performance when the maximum desired training budget is
    not known a priori. This is desirable if increasing the training budget
    will continue to yield an improved estimate of the performance of the
    configuration. However, it should also be the case a reasonable estimate
    of the configuration's performance can be obtained with a very small
    training budget.

    Parameters
    ----------
    time_attr (str): A training result attr to use for comparing time.
      Note that you can pass in something non-temporal such as
      `training_iteration` as a measure of progress, the only requirement
      is that the attribute should increase monotonically.
     metric (str): The training result objective value attribute. Stopping
      procedures will use this attribute. If None but a mode was passed,
      the `ray.tune.result.DEFAULT_METRIC` will be used per default.
    mode (str): One of {min, max}. Determines whether objective is
      minimizing or maximizing the metric attribute.
    max_t (float): max time units per trial. Trials will be stopped after
      max_t time units (determined by time_attr) have passed.
    grace_period (float): Only stop trials at least this old in time.
      The units are the same as the attribute named by `time_attr`.
    reduction_factor (float): Used to set halving rate and amount. This
      is simply a unit-less scalar.
    brackets (int): Number of brackets. Each bracket has a different
      halving rate, specified by the reduction factor.
    """
    def __init__(self, *args, grace_period=1, **kwargs):
        super().__init__(*args, grace_period=grace_period, **kwargs)
        self._bracket_released = []
        for i in range(len(self._brackets)):
            self._bracket_released.append(defaultdict(list))

        # Replace the depth-first brackets from the original variant of
        # ASHA with some breadth-first ones.
        self._brackets = [
            _BreadthFirstBracket(grace_period, self._max_t, self._reduction_factor, s)
            for s in range(len(self._brackets))
        ]

    def choose_trial_to_run(self, trial_runner: "trial_runner.TrialRunner"):
        # TODO: Now that we actually have our own version of the bracket class,
        # this really ought to be folded into it.
        trial_to_run = None
        not_ready_to_start = set([])
        # Pick the next run from a random bracket
        for bracket_idx in np.random.permutation(range(len(self._brackets))):
            bracket = self._brackets[bracket_idx]
            bracket_released = self._bracket_released[bracket_idx]
            # First, select a trial that has been released already and is not
            # yet at the next milestone
            for milestone_idx in range(1, len(bracket._rungs)): 
                _, recorded_upper = bracket._rungs[milestone_idx-1]
                milestone_lower, _ = bracket._rungs[milestone_idx]
                released_lower = bracket_released[milestone_lower]
                logging.debug('\n{} - released: {}; recorded: {}'.format(milestone_lower, released_lower, recorded_upper))
                for trial_id in released_lower:
                    if trial_id not in recorded_upper:
                        is_ready, trial_to_run = self._is_ready_to_run(trial_id, trial_runner)
                        if is_ready:
                            break
                        else:
                            trial_to_run = None
                if trial_to_run is not None:
                    break
            if trial_to_run is not None:
                logging.debug('\nTrial to run ({}) came from phase 1!'.format(trial_to_run.trial_id))
            # If we still don't have a trial to run, check to see if any are at
            # a milestone and can be released
            if trial_to_run is None:
                for milestone, recorded in bracket._rungs:
                   already_released = bracket_released[milestone]
                   logging.debug('\n{}---> already_released: {} - recorded: {} - rf: {}'.format(milestone, already_released, recorded, bracket.rf))
                   if len(already_released) < int(len(recorded)/bracket.rf):
                       logging.debug('\nWe can release a new run!' + '*'*20)
                       # This bracket rung has enough recorded results to release
                       # another configuration
                       for trial_id in reversed(sorted(recorded, key=lambda k: recorded[k])):
                           logging.debug('\n' + '-'*60)
                           logging.debug('\nTrying to release {}'.format(trial_id))
                           if trial_id in already_released:
                               logging.debug('\nIt was already released')
                               not_ready_to_start.add(trial_id)
                               continue
                           # This trial can be released if it is possible to run it
                           # right now
                           is_ready, trial_to_run = self._is_ready_to_run(trial_id, trial_runner)
                           if is_ready:
                               already_released.append(trial_id)
                               break
                           else:
                               if trial_to_run is not None:
                                   logging.debug('\nIt was not ready to run, status was: {}'.format(trial_to_run.status))
                               else:
                                   logging.debug('\nIt was not ready to run, because it could not be found.')
                               not_ready_to_start.add(trial_id)
                               trial_to_run = None
                       if trial_to_run is not None:
                           break
            if trial_to_run is not None:
                logging.debug('\nTrial to run ({}) came from phase 2!'.format(trial_to_run.trial_id))
        # None of the other trials are ready to be released, so we start a new
        # one instead
        if trial_to_run is None:
            for trial in trial_runner.get_trials():
                if (trial.trial_id not in not_ready_to_start
                        and trial.status in [Trial.PENDING]
                        and trial_runner.has_resources(trial.resources)):
                    trial_to_run = trial
                    break
            if trial_to_run is not None:
                logging.debug('\nTrial to run ({}) came from phase 3!'.format(trial_to_run.trial_id))
                 
        return trial_to_run
                           
    def _is_ready_to_run(self, trial_id, trial_runner):
        """_is_ready_to_run
        
        Gets the trial corresponding to the ID and checks if it is ready to be
        run.
        """
        for trial in trial_runner.get_trials():
            if trial.trial_id == trial_id:
                return (trial.status in [Trial.PAUSED] 
                        and trial_runner.has_resources(trial.resources)), trial
        return False, None

class _BreadthFirstBracket(_Bracket):
    def on_result(self, trial: Trial, cur_iter: int,
                  cur_rew: Optional[float]) -> str:
        message = ['*'*60, 'current iteration: {}'.format(cur_iter)]
        action = TrialScheduler.CONTINUE
        for milestone, recorded in self._rungs:
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                message.append('Found matching milestone: {}'.format(milestone))
                message.append('Currently recorded: {}'.format(recorded))
                action = TrialScheduler.PAUSE
                if cur_rew is None:
                    logger.warning("Reward attribute is None! Consider"
                                   " reporting using a different field.")
                else:
                    recorded[trial.trial_id] = cur_rew
                break
        message.append('action = {}'.format(action))
        logging.debug('{}\n'.format('\n'.join(message)))
        return action

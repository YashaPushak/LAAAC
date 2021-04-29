import time
import logging

import numpy as np

try:
    from ray.tune import Stopper
    ray_available = True
except:
    ray_available = False

if ray_available:
    class BudgetStopper(Stopper):
        def __init__(self, config_budget=float('inf'), column_name='current_fidelity_budget', count_if_true_column=None,
                     log_location=None, log_level=logging.INFO, log_formatter=None):
            self._budget = 0
            self._max_budget = config_budget
            self._column_name = column_name
            self._cit = count_if_true_column
            self._logger = logging.getLogger('BudgetStopper')
            if log_location is not None:
                fh = logging.FileHandler(log_location)
                fh.setLevel(log_level)
                if log_formatter is not None:
                    fh.setFormatter(log_formatter)
                self._logger.addHandler(fh)
            self._logger.info('Initialized Budget Stopper')
    
        def __call__(self, trial_id, result):
            self._logger.debug(f'count_if_true: {self._cit}')
            if self._cit is not None:
                self._logger.debug(f'result[count_if_true] = {result[self._cit]}')
            if self._cit is None or result[self._cit]:
                self._budget += np.sum(result[self._column_name])
                self._logger.debug(f'Adding {result[self._column_name]} to exhausted budget')
                self._logger.debug(f'Exhausted budget is now {self._budget}/{self._max_budget}')
            return False
    
        def stop_all(self):
            stop = self._budget >= self._max_budget
            if stop:
                self._logger.info(f'Sending signal to stop configuration because the budget is '
                                  f'{self._budget} >= {self._max_budget}')
            else:
                self._logger.debug(f'Not ready to stop yet: {self._budget} < {self._max_budget}')
            return stop

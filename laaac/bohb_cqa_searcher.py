import logging

import numpy as np

try:
    from hpbandster.core.base_config_generator import base_config_generator
    bohb_available = True
except:
    bohb_available = False

from .cqa_searcher import CoreCQASearcher

LOGGER = logging.getLogger('config_generator_CQAHB')


class CQAHB:
    def __init__(self, *args, **kwargs):
        raise ImportError('The hyperbandster package must be installed first. Please install it to continue.')

if bohb_available:
    class CQAHB(base_config_generator):
        def __init__(self, **kwargs):
            """CQAHB
    
            Initializes the CQA Search algorithm used by CQAHB. See
            laaac.cqa_searcher.CQASearher for details.
            """
            LOGGER.debug(f'Intializating CQAHB configuration generator\n'
                         f'kwargs={kwargs}')
            super().__init__()
            LOGGER.debug('Called super().__init__')
            
            if 'metric' not in kwargs:
                kwargs['metric'] = 'loss'
            self._cqa = CoreCQASearcher(**kwargs)
            LOGGER.debug('Initialized CQASearcher')
            self._next_trial_id = 0
            self._config_to_trial_id = {}
    
        def get_config(self, *args):
            trial_id = self._next_trial_id
            self._next_trial_id += 1
            config = self._cqa.suggest(trial_id)
            self._config_to_trial_id[_immutable(config)] = trial_id
            return config, {}
    
        def new_result(self, job, update_model=True):
            super().new_result(job)
            # Create a result in the format expected by the CQA searcher      
            if job.result is None:
                loss = np.nan  # Will be overwritten later by the crash_score
            else:
                loss = job.result['loss'] if np.isfinite(job.result['loss']) else np.inf
            result = {}
            result[self._cqa._metric] = loss
            result[self._cqa._budget] = job.kwargs['budget']
            # look up the trial ID
            trial_id = self._config_to_trial_id[_immutable(job.kwargs['config'])]
            # report the result
            self._cqa.on_trial_result(trial_id, result, update_model)

def _immutable(config):
    """_immutable
    Returns an immutable representation of a configuration that can be used as
    a key in a dict.

    Parameters
    ----------
    config : dict | Configuration
      The configuration to convert.

    Returns
    -------
    tuple of tuples
      The immutable representation of the configuration.
    """
    # In case the configuration was a Configuration instead of a dict.
    config = dict(config)
    return tuple((key, config[key]) for key in sorted(config.keys()))

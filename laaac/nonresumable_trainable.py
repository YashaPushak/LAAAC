import copy
import pickle

import numpy as np

def nonresumable_trainable_factory(UserTrainable, grace_period,  max_t, reduction_factor=2,
                                   budget_transformer=lambda b: b,
                                   time_attr='number_of_step_calls', **kwargs):
    """nonresumable_trainable_factory

    A wrapper for using target algorithm runners that cannot be resumed, as
    ASHA assumes, such that evaluations only happen when the current step
    is a milestone.

    Parameters
    ----------
    UserTrainable : <class 'ray.tune.Trainable(ray.tune.Trainable)'>
      Your custom Trainable class that cannot have training resumed.
    grace_period : float
      Must correspond to the grace period provided to ASHA.
    max_t : float
      Must correspond to the max_t provided to ASHA.
    reduction_factor : float
      Must correspond to the reduction factor provided to ASHA.
    budget_transformer : callable
      A function used to transform the budget from an integer number of steps
      to the fidelity budget to be provided to your trainable. Note that if
      provided, then the grace_period and max_t used here and in ASHA must 
      be represented in the inverse transform of your budget transformer.
      For example, if your fidelity budget is the number of training iterations
      and it can range between 500 to 10000 in steps of 500, then set 
      grace_period = 1, max_t = 20 and budget_transformer = lambda b: b*500.
    time_attr : str
      The name of the attribute which the non-resumable trainable will write
      into the result of your trainable so that ASHA knows how many calls to
      step have been performed for this trainable. That is, you should tell
      ASHA that the time_attr of the trainable corresponds to this string.
    kwargs : dict
      All other key-work arguments are ignored. This is a convenience so that
      you can use a single dict to set the parameters of the function and 
      the parameters of ASHA.

    Returns
    -------
    <class 'NonResumableTrainable(UserTrainable(ray.tune.Trainable))'>
      A class that implements the ray.tune.Trainable interface, but that
      only evaluates the UserTrainable at milestones, and otherwise returns
      the last-seen result from the UserTrainable.
    """ 
    class NonResumableTrainable(UserTrainable):
        """
        Wraps calls to the UserTrainable so that the UserTrainable step
        function is only called at milestones. For all other steps, the value
        of the last step function call to the user trainable is returned 
        instead. This allows for UserTrainbles to be used that are not natively
        able to pause and resume their training. For maximum effectiveness,
        this class should be used in combination with a successive-halving-
        algorithm-based method, using the same grace period, reduction factor
        and max t as the successive halving algorithm.
        """
        def __init__(self, *args, **kwargs):
            MAX_RUNGS = int(np.log(max_t/grace_period)/np.log(reduction_factor) + 1)
            self.__milestones = [grace_period*reduction_factor**k
                                 for k in reversed(range(MAX_RUNGS))]
            # Remove the grace period, because we handle the first step as a 
            # special case
            self.__milestones.remove(grace_period)
            self.__grace_period = grace_period
            self.__current_step = 0
            self.__last_result = None
            self.__time_attr = time_attr
            self._set_budget(grace_period)
            super().__init__(*args, **kwargs)

        def setup(self, *args, **kwargs):
            self.__current_step = 0
            self.__last_result = None
            self._set_budget(grace_period)
            super().setup(*args, **kwargs) 

        def _set_budget(self, budget):
            self.training_budget = budget_transformer(budget)

        def step(self, *args, **kwargs):
            self.__current_step += 1
            if self.__current_step in self.__milestones or self.__last_result is None:
                self._set_budget(max(self.__current_step, self.__grace_period))
                self.__last_result = super().step(*args, **kwargs)
                self.__last_result['fresh'] = True
            else:
                self.__last_result['fresh'] = False
            result = copy.deepcopy(self.__last_result)
            # Overwrite the time attribute with the current step so that
            # ASHA knows how to properly track the budget spent so far
            result[self.__time_attr] = self.__current_step
            return result
       
        def save_checkpoint(self, checkpoint_dir, checkpoint_dict=None):
            path = super().save_checkpoint(checkpoint_dir, checkpoint_dict)
            data = {'current_step': self.__current_step,
                    'last_result': self.__last_result,
                    'path': path}
            if checkpoint_dict is not None:
                checkpoint_dict[path] = (checkpoint_dict[path], data)
            else:
                path = '{}_nonresumable'.format(path)
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            return path

        def load_checkpoint(self, checkpoint_path, checkpoint_dict=None):
            if checkpoint_dict is not None:
                user_data, data = checkpoint_dict[checkpoint_path]
                checkpoint_dict[checkpoint_path] = user_data
            else:
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                checkpoint_path = data['path']
            self.__current_step = data['current_step']
            self.__last_result = data['last_result']
            super().load_checkpoint(checkpoint_path, checkpoint_dict)
            return checkpoint_path

    return NonResumableTrainable

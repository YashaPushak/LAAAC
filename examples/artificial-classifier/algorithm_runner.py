import time

import numpy as np

from laaac.abstract_runner import AbstractRunner


class TargetAlgorithmRunner(AbstractRunner):

    def __init__(self, *args, **kwargs):
        pass

    def perform_run(self, parameters, instance, seed, cutoff, **kwargs):
        """perform_run

        Performs a the validation error from a simulated run of xgboost.

        Parameters
        ----------
        parameters : dict
          The hyper-parameter configuration to evaluate.
        instance : str
          The name of the instance (here: cross-validation "fold") on
          which to evaluate the configuration.
        seed : int
          The random seed to use for the simulated xgboost run.
        cutoff : float
          The budget to use for the training run. GPS assumes this
          is measured in seconds, but in fact you could provide any
          meaningful value measured in any units to the scenario, which 
          will be passed to your algorithm here.
        **kwargs
          Additional fields not needed for this example.
         
        Returns
        -------
        result : str
          Should be one of 'SUCCESS', 'TIMEOUT', or 'CRASHED'.
        runtime : float
          The running time used by your target algorithm to perform the run.
          If optimizing for solution quality, this is still used for 
          CPU-time-based configuration budgets.
        solution_quality : float
          The solution quality obtained by your target algorithm on this 
          this instance. If optimizing for running time, this field is
          ignored by GPS (but still required).
        miscellaneous : str
          Miscellaneous data returned by your target algorithm run. This 
          must be comma-free, but otherwise will be ignored by GPS.
        """       
        # Default values to overwrite if the run is successful.
        result = 'CRASHED'
        runtime_observed = 0
        error_observed = np.inf
        miscellaneous = 'out of bounds'
        
        instance_seed = hash(instance)
        x0 = parameters['x0']
        x1 = parameters['x1']
   
        # Let's assume that we're optimizing two parameters of a machine learning
        # classifier for binomial classification. Let's further assume that the
        # errors are binomially distributed (this is a simplification of reality,
        # see Emil and Tamer, 2013 "Some statistical aspects of binary measuring
        # systems"). We will therefore sample from a binomial distribution with
        # a probability, p, of making errors. We will then choose p as a function
        # of x0 and x1. A single call to our machine learning algoirthm given a
        # particular "cross validation fold" (instance number) will correspond to
        # a single random sample from this binomial distribution with, say 100
        # instances and we will then count the number of times that the model made
        # an error.
    
        # Let the probability of a failure correspond to a quadratic function that
        # is minimized by (5, 5). We'll make this function so that the features do
        # interact -- that is, the quadratic function will be squished upwards
        # along the axis x0 = 10 - x1.
        def p_failure(x0, x1):
            def _p(x0, x1):
                return (x0 + x1 - 10)**2 + (x0 - 5)**2 + (x1 - 5)**2
            # x0 and x1 should be in the range [0, 10], so we normalize the
            # function by the worse solution quality obtained at (0,0) or (10, 10)
            # and then we divide by 2 because we assume that any good ML system
            # should do no worse than random guessing.
            return _p(x0, x1)/_p(0, 0)/2
        # Calculate the probability of an error given these parameter settings
        deterministic_p = p_failure(x0, x1)
    
        # Let us further assume that there is a small amount of noise due to the
        # particular fold used for training, which we will model using a truncated
        # normal distribution. We add x0 and x1 to the instance seed because we
        # expect that changing the parameter value by a tiny amount to have an
        # equivalent effect as if we had changed the random seed.
        np.random.seed((instance_seed + hash(x0) + hash(x1) + 12345) % 4294967294)
        fold_p = np.random.normal(deterministic_p, 0.01)
        # Keep p in [0, 1]
        fold_p = min(max(fold_p, 0), 1)
    
        # The number of test instances
        n = 1000
    
        # Finally, sample from the binomial distribution
        np.random.seed((seed + hash(x0) + hash(x1) + 54321) % 4294967294)
        if instance != 'test':
            n_errors = 1.0*np.random.binomial(n, fold_p)/n
        else:
            n_errors = deterministic_p
    
        # Let's just make the simple assumption that these running times are normally
        # distributed.
        runtime = max(np.random.normal(5, 1), 0.1)
    
        result = 'SUCCESS'
        if cutoff is not None and runtime > cutoff:
            runtime = cutoff
            result = 'TIMEOUT'
            # It's up to you to return an appropriate status message if your
            # algorithm terminates due to a running time cutoff. If you have
            # an anytime optimization algorithm, you should just return
            # "SUCCESS" and the final solution quality found. However, if
            # you were unable to find any solution to your problem within
            # the cutoff, then returning 'TIMEOUT' is appropriate.
            # We're going to assume this is the case here.
            n_errors = 1
            # Note that GPS will still use this objective function value
            # rather than treating this similarly to if it were a crashed
            # run like it would if we were optimizing for running times.
    
        misc = ('Miscellaneous extra data from the run (ignored by GPS) '
                '- deterministic probability of errors {0:.6f}'
                ''.format(deterministic_p))

        return result, runtime, n_errors, misc
                

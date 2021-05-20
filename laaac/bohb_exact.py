import logging
import traceback
from contextlib import contextmanager

import numpy as np
from scipy.optimize import dual_annealing

import ConfigSpace

try:
    from hpbandster.optimizers.config_generators.bohb import BOHB
    bohb_available = True
except:
    bohb_available = False    



LOGGER = logging.getLogger('BOHB-Exact')


@contextmanager
def exception_logging():
    try:
        yield
    except Exception as e:
        LOGGER.error('An exception occured.', exc_info=e)
        raise e
    finally:
        pass



class BOHBExact:
    def __init__(self, *args, **kwargs):
        raise ImportError('the hpperbandster package must be available first. Please install it.')


if bohb_available:
    class BOHBExact(BOHB):
    
        @exception_logging()
        def get_config(self, budget):
            """
            Function to sample a new configuration
            This function is called inside Hyperband to query a new configuration
            Parameters:
            -----------
            budget: float
              the budget for which this configuration is scheduled
            returns: config
              should return a valid configuration
            """
            self.logger.debug('start sampling a new configuration.')
            
            sample = None
            info_dict = {}
            
            # If no model is available, sample from prior
            # also mix in a fraction of random configs
            if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
                sample =  self.configspace.sample_configuration()
                info_dict['model_based_pick'] = False
            
            best = np.inf
            best_vector = None
    
            if sample is None:
                try:
                    # sample from largest budget
                    budget = max(self.kde_models.keys())
                    
                    l = self.kde_models[budget]['good'].pdf
                    g = self.kde_models[budget]['bad' ].pdf
    
                    # The locations of the categorical hyper-parameters
                    cats = self.vartypes > 0
    
                    def aquisition_function(x):
                        # Clip and round categorical hyper-parameters to the indices
                        # of their values
                        x[cats] = np.clip(np.rint(x[cats]), 0, self.vartypes[cats])
                        # Get the objective function value
                        val = max(1e-32, g(x))/max(l(x),1e-32)
                        if not np.isfinite(val):
                            # right now, this happens because a KDE does not contain
                            # all values for a categorical parameter. This cannot be
                            # fixed with the statsmodels KDE, so for now, we are just
                            # going to evaluate this one if the good_kde has a finite
                            # value (there is no config with that value in the bad
                            # kde, so it shouldn't be terrible).
                            if not np.isfinite(l(vector)):
                                val = np.inf
                        return val
    
                    lower_bounds = np.zeros((len(self.vartypes), 1))
                    upper_bounds = np.ones((len(self.vartypes), 1))
                    upper_bounds[cats, 0] = self.vartypes[cats]
                    bounds = np.append(lower_bounds, upper_bounds, axis=1)
    
                    res = dual_annealing(aquisition_function, bounds)
                    best_vector = res.x if res.success else None
                    best = aquisition_function(best_vector)
                    self.logger.debug(f'Dual annealing optimization Result: {res}')
                    random_scores = []
                    for _ in range(50):
                       x = np.random.random(len(best_vector))
                       for cat in np.where(cats)[0]:
                           x[cat] = np.random.randint(*bounds[cat])
                       random_scores.append(aquisition_function(x))
                    self.logger.debug(f'Random aquisition scores: {np.quantile(random_scores, [0, 0.25, 0.5, 0.75, 1])}')
                    
                    if best_vector is None:
                        self.logger.debug("Sampling based optimization with %i samples failed -> using random configuration"%self.num_samples)
                        sample = self.configspace.sample_configuration().get_dictionary()
                        info_dict['model_based_pick']  = False
                    else:
                        self.logger.debug('best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                        for i, hp_value in enumerate(best_vector):
                            if isinstance(
                                self.configspace.get_hyperparameter(
                                    self.configspace.get_hyperparameter_by_idx(i)
                                ),
                                ConfigSpace.hyperparameters.CategoricalHyperparameter
                            ):
                                best_vector[i] = int(np.rint(best_vector[i]))
                        sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()
                        
                        try:
                            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                                        configuration_space=self.configspace,
                                        configuration=sample
                                        )
                            info_dict['model_based_pick'] = True
                    
                        except Exception as e:
                            self.logger.warning(("="*50 + "\n")*3 +\
                                    "Error converting configuration:\n%s"%sample+\
                                    "\n here is a traceback:" +\
                                    traceback.format_exc())
                            raise(e)
                    
                except:
                    self.logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration"%(self.num_samples, traceback.format_exc()))
                    sample = self.configspace.sample_configuration()
                    info_dict['model_based_pick']  = False
            try:
                sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                    configuration_space=self.configspace,
                    configuration=sample.get_dictionary()
                ).get_dictionary()
            except Exception as e:
                self.logger.warning("Error (%s) converting configuration: %s -> "
                                    "using random configuration!",
                                    e,
                                    sample)
                sample = self.configspace.sample_configuration().get_dictionary()
            self.logger.debug('done sampling a new configuration.')
            self.logger.debug(f'Configuration being suggested:\n{sample}')
            return sample, info_dict

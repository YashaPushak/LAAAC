import copy

class AbstractRunner:
    def perform_run(self, parameters, instance, instance_specifics, seed, 
                    cutoff, run_length, run_id, temp_dir='.'):
        """perform_run

        Performs the specified run of the target algorithm and returns the
        results from the run.

        Parameters
        ----------
        parameters : dict
          A dictionary mapping parameter names to values. This defines the
          parameter configuration to be evaluated.
        instance : str
          The name of the instance on which the configuration is to be 
          evaluated. This will correspond directly to one of the lines 
          defined in your instance file.
        instance_specifics : str
          Not currently supported. This value will always be "0" when your
          target algorithm is called.
        seed : int
          The random seed to be used by your target algorithm.
        cutoff : float | None
          A running time cutoff to be used by your target algorithm. Note
          that you must enforce this cutoff in your target algorithm or 
          wrapper, it will not be done for you. The cutoff time is in seconds.
        run_length : int
          Not currently supported. This value will always be 0.
        run_id : str
          A randomly generated string that you may optionally use to uniquely
          identify this run of your target algorithm.
        temp_dir : str
          Note that GPS-ASHA and SCQA-ASHA don't current do anything
          intelligent here, so this will simply be the current working
          directory for the experiment. If you need to create temporary files
          you should probably create them in a directory that makes more sense
          like /temp/. However, this is left in for backwards-compatability.
                  
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
        pass
          
    def _get_command(self, wrapper, parameters, instance, instance_specifics, 
                     seed, cutoff, run_length, run_id, temp_dir):
        return ('target_runner.perform_run(parameters={parameters},\n'
                '                          instance=\'{instance}\',\n'
                '                          instance_specifics=\'{instance_specifics}\',\n'
                '                          seed={seed},\n'
                '                          cutoff={cutoff},\n'
                '                          run_length={run_length},\n'
                '                          run_id=\'{run_id}\',\n'
                '                          temp_dir=\'{temp_dir}\')'
                ''.format(parameters=parameters, 
                          instance=instance,
                          instance_specifics=instance_specifics,
                          seed=seed, cutoff=cutoff, run_length=run_length, 
                          run_id=run_id, temp_dir=temp_dir))
 

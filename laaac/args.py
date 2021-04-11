import inspect
import argparse
import os

from laaac import helper


class ArgumentParser:
    """ArgumentParser

    The ArgumentParser can parse command line arguements, scenario file
    arguments and global default arguments to initialize the parameters of the optimizer.
    """
    def __init__(self):
        self.setup_arguments = {
           ('--scenario-file', '--scenario'): {
                'help': 'The scenario file (and location) that defines what settings are used for optimizer.',
                'type': str},
           ('--experiment-dir', '--exec-dir', '-e', '--run-directory'): {
                'help': 'The root directory from which experiments will be run. By default, this is the '
                        'current working directory. The optimizer will change to this directory prior to running, '
                        'this means that if relative paths are specified for any other files or directories '
                        'then they must be given relative to your experiment directory.',
                'type': _validate(str, 'The experiment directory must be a valid directory', lambda x: helper.isDir(x))},
            ('--output-dir', '--output-directory', '--out-dir', '--log-location'): {
                'help': 'The directory where output will be stored. The actual directory for a particular '
                        'optimizer run with run ID run_id will be stored in {experiment-dir}/{output-dir}/{optimizer}/{run_id}',
                'type': str},
            ('--verbose', '--verbosity', '--log-level', '-v'): {
                'help': 'Controls the verbosity of the output. Set of 0 for warnings only. Set to '
                        '1 for more informative messages. And set to 2 for debug-level messages. The '
                        'default is 1.',
                'type': _validate(int, 'The verbosity must be in [0, 1, 2]', lambda x: 0 <= x <= 2)},
            ('--optimizer', ): {
                'help': 'Determines what kind of optimizer is used to explore the configuration space. '
                        'Available options are: "GPS", which uses a golden-section-search-based line-search '
                        'procedure along randomly chosen vectors of the configuration space; "CQA", which '
                        'sequentially fits convex quadratic approximations of the loss landscape that '
                        'under-estimate observed losses and uses the resulting models to guide the search '
                        'process.',
                'type': _validate(str, 'The optimizer must be one of "GPSLS" or "CQA"', lambda x: str(x).lower() in ['gpsls', 'cqa'])},
        }
        
        self.scenario_arguments = {
            ('--pcs-file', '--param-file', '--p'): {
                'help': 'The file that contains the algorithm parameter configuration space in PCS format. '
                        'The optimizers support the same syntax used for BOHB, SMAC and ParamILS.',
                'type': str},
            ('--instance-file', '--instances', '-i'): {
                'help': 'The file (and location) containing the names of the instances to '
                        'be used to evaluate the target algorithm\'s configurations. If no '
                        'instances are to be used, set to "None".',
                'type': str},
            ('--algo', '--algo-exec', '--algorithm', '--wrapper'): {
                'help': 'This should be the name of the python file that implements the target-'
                        'algorithm interface.',
                'type': str},
            ('--algo-cutoff-time', '--target-run-cputime-limit', '--cutoff-time', '--cutoff'): {
                'help': 'The wallclock time limit for an individual target algorithm run, in seconds. If set to '
                        'zero, no cutoff time will be used.',
                'type': _validate(float, 'The cutoff time must be a non-negative number', lambda x: float(x) >= 0)},
            ('--config-budget', '--configuration-fidelity-budget', '--runcount-limit', '--total-num-runs-limit',
             '--num-runs-limit', '--number-of-runs-limit'): {
                'help': 'Limits the sum of the fidelity budget used to evaluate the configurations. For backwards '
                        'compatability, this contains aliases for runcount limit, for which the name only makes sense '
                        'if lower fidelity evaluations of a configuration correspond to fewer independent runs of the '
                        'target algorithm. In practise, if the measure of fidelity corresponds to, e.g., training '
                        'iterations of an algorithm, then use this setting to limit the sum of the training iterations '
                        'across all independent runs of the target algorithm. Either this or the wallclock limit must '
                        'be less than the maximum integer value. The default is the maximum integer value.',
                'type': _validate(int, 'The configuration budget must be a positive integer', lambda x: int(x) > 0)},
            ('--wallclock-limit', '--runtime-limit'): {
                'help': 'Limits the total wall-clock time used by the optimizer, in seconds. Either this, the runcount  or the CPU '
                        'time limit must be less than the maximum integer value. The default is the maximum integer '
                        'value.',
                'type': _validate(float, 'The wall-clock time must be a positive, real number', lambda x: float(x) > 0)},
            ('--cpus-per-trial', ): {
                'help': 'The number of CPUs that will be allocated to each target algorithm run.',
                'type': _validate(int, 'The number of CPUs per trial must be a positive integer.', lambda x: int(x) > 0)},
            ('--gpus-per-trial', ): {
                'help': 'The number of GPUs that will be allocated to each target algorithm run.',
                'type': _validate(int, 'The number of GPUs per trial must be a non-negative integer.', lambda x: int(x) >= 0)},
            ('--cluster-ip-address', '--cluster' ): {
                'help': 'Can be used to specify the IP address of a ray-cluster to connect to and use for parallelism.'
                        'set to None (the default) to run locally. Note that the ray-cluster must already be running '
                        'and configured for this to work if not None.',
                'type': str},
            ('--grace-period', '--min-fidelity', '--min-budget', '--minimum-runs', '--min-runs'): {
                'help': 'The minimum fidelity budget that a configuration must be evaluated on before it can '
                        'be eliminated.',
                'type': _validate(float, 'The grace period must be a positive number.', lambda x: float(x) > 0)},
            ('--max-fidelity', '--max-t', '--max-runs'): {
                'help': 'The largest fidelity budget with which to evaluate any single configuration.',
                'type': _validate(float, 'The maximum fidelity must be a positive number.', lambda x: float(x) > 0)},
            ('--fidelity-step', '--budget-step', '--fidelity-budget-step'): {
                'help': 'The amount by which the fidelity of the budget should be incremented at each call to '
                        'the target algorithm runner. Ignored if objective is not "final".',
                'type': _validate(float, 'The fidelity budget step must be a positive number', lambda x: float(x) > 0)},
            ('--integer-fidelity', '--integer-fidelity-budget', '--integer-budget'): {
                'help': 'If True, the fidelity budget will be rounded and cast to an integer after incrementing it by '
                        'the budget step.',
                'type': _validate(_to_bool, 'The integer fidelity parameter must be "True" or "False".')},
            ('--crashed-loss', '--crashed-score', '--crash-score', '--crash-loss'): {
                'help': 'The loss or solution quality score to assign to configurations which do not '
                        'properly terminate or report valid scores. This should ideally be larger than '
                        'the largest value your algorithm can return for the loss. If the optimizer is '
                        'CQA, then this must be finite, and ideally should not be more than one order '
                        'of magnitude other than the largest value return by your algorithm or any '
                        'crashed results may substantially biase the fitted models.',
                'type': float},
            ('--objective',): {
                'help': 'Must be one of "mean" or "final". If "mean", the optimizer will seek to optimize '
                        'the mean loss reported for each fidelity budget. This should always be preferred '
                        'when the measure of fidelity corresponds to, e.g., the number of independent runs '
                        'of the target algorithm on different cross-validation folds or problem instances. '
                        'However, if the measure of fidelity corresponds to target algorithm training '
                        'iterations, then "final" should be preferred.',
                'type': _validate(str, 'The objective must be either "mean" or "final"',
                                  lambda x: x.lower() in ['mean', 'final'])},
        }
        
        self.groups_in_order = ['Setup Arguments', 'Scenario Arguments']
        self.argument_groups = {'Setup Arguments': self.setup_arguments,
                                'Scenario Arguments': self.scenario_arguments}
        self.group_help = {'Setup Arguments': 'These are general optimizer arguments that are used to set up '
                                              'the optimizer run.',
                           'Scenario Arguments': 'These arguments define the scenario-specific '
                                                 'information.'}
        # Location of the source code directory
        package_directory = os.path.dirname(os.path.realpath(inspect.getfile(inspect.currentframe())))
        # File with hard-coded default values for all (optional) parameters
        self.defaults = '{}/.defaults.txt'.format(package_directory)

    def parse_command_line_arguments(self):
        """parse_command_line_arguments
    
        Parses the command line arguments.
    
        Returns
        -------
        arguments: dict
            A dictionary containing the parsed arguments.
        """
        parser = argparse.ArgumentParser()
        for group_name in self.argument_groups:
            group = parser.add_argument_group(group_name)
            for arg in self.argument_groups[group_name]:
                group.add_argument(*_get_aliases(arg), dest=_get_name(arg), **self.argument_groups[group_name][arg])
        # Parse the command line arguments and convert to a dictionary
        args = vars(parser.parse_args())
        keys = list(args.keys())
        # Remove everything that is None so that we know to replace those values with scenario file arguments
        # instead.
        for arg in keys:
            if args[arg] is None:
                del args[arg]
        return args
   
    def parse_file_arguments(self, scenario_file, override_arguments={}):
        """parse_file_arguments
    
        Reads in the scenario file arguments, over-writes any of them with their
        override counterparts (for example, defined on the command line), if 
        applicable, and then saves them.
        """ 
        parsed_arguments = {}
        skipped_lines = []
        with open(scenario_file) as f_in:
            for line in f_in:
                # Remove any comments
                line = line.split('#')[0]
                # Strip whitespace
                line = line.strip()
                # Skip empty lines
                if len(line) == 0:
                    continue
                key = line.split('=')[0].strip()
                value = '='.join(line.split('=')[1:]).strip()
                found = False
                # Check for a match in any of the argument types
                for group in self.argument_groups: 
                    for argument in self.argument_groups[group]:
                        if '--{}'.format(key) in _get_aliases(argument) or '-{}'.format(key) in argument:
                            # We found a match, store it under the argument's proper name, convert the
                            # value to it's proper type and raise an exception if it is invalid.
                            parsed_arguments[_get_name(argument)] \
                                = self.argument_groups[group][argument]['type'](value)
                            found = True
                            continue
                if found:
                    continue
                if not found:
                    skipped_lines.append(line)
        # Overwrite any argument definitions, as needed 
        for argument in override_arguments:
            parsed_arguments[argument] = override_arguments[argument]

        return parsed_arguments, skipped_lines        

    def parse_arguments(self):
        """parse_arguments
        Parse the command line arguments, then, if provided, parse the 
        arguments in the scenario file. Then adds default values for
        paramaters without definitions. Finally, validates all argument
        definitions, checks that needed files and directories exist, and then
        checks to make sure that all required arguements received definitions.
        
        Returns
        -------
        arguments : dict
            A dictionary mapping all arguments to definitions.
        skipped_lines : list of str
            A list of all non-comment lines in the scenario file that were
            skipped.
        """
        skipped_lines = []
        # First parse the command line arguments
        arguments = self.parse_command_line_arguments()
        # If a scenario file was provided, parse the arguments from it
        if 'scenario_file' in arguments:
            # If an experiment directory is specified, we will change to that directory
            experiment_dir = arguments['experiment_dir'] if 'experiment_dir' in arguments else '.'
            with helper.cd(experiment_dir):
                try:
                    arguments, skipped_lines = self.parse_file_arguments(arguments['scenario_file'], arguments) 
                except IOError:
                    raise IOError("The scenario file '{}' could not be found from within the "
                                  "current working directory '{}' (which is the experiment directory, "
                                  "if one was specified on the command line)."
                                  "".format(arguments['scenario_file'], os.getcwd()))
        # Finally, load the default values of all the parameters (that make sense to be shared)
        arguments, _ = self.parse_file_arguments(self.defaults, arguments)
        # Check that all parameters have defintions (optional parameters not specified by the
        # user will have already been included with default values)
        self._validate_all_arguments_defined(arguments)
        # Make sure all of the files and directories can be found
        _validate_files_and_directories(arguments)
        # Make sure the budget was set
        _validate_budget(arguments)

        # Save the data for later
        self.parsed_arguments = arguments

        return arguments, skipped_lines

    def _validate_all_arguments_defined(self, arguments):
        missing = []
        # iterate over all arguments
        for group in self.argument_groups: 
            for argument in self.argument_groups[group]:
                name = _get_name(argument)
                if name not in arguments:
                    missing.append(name)
        # The scenario file is the only argument that is *truely* optional
        if 'scenario_file' in missing:
            missing.remove('scenario_file')
        if len(missing) > 0:
            raise TypeError('The optimizer was missing definitions for the following required arguments: {}'
                            ''.format(missing))       

    def create_scenario_file(self, scenario_file, arguments):
        """create_scenario_file

        Creates a scenario file with the specified name and arguments.
        """
        with open(scenario_file, 'w') as f_out:
            for group in self.argument_groups:
                f_out.write('# {}\n'.format(group))
                f_out.write('# {}\n'.format('-'*len(group)))
                for argument in self.argument_groups[group]:
                    name = _get_name(argument)
                    # Of course it doesn't really make sense to save
                    # the name of the file in the file...
                    if name == 'scenario_file':
                        continue
                    f_out.write('{} = {}\n'.format(name, arguments[name]))
            f_out.write('\n')


def _get_name(names):
    name = names[0] if isinstance(names, tuple) else names
    name = name[2:] if len(name) > 2 else name[1]
    return name.replace('-','_')

 
def _validate(types, message=None, valid=lambda x: True):
    if not isinstance(types, tuple):
        types = (types, )
    def _check_valid(input_):
        valid_type = False
        for type_ in types:
            try:
                input_ = type_(input_)
                valid_type = True
            except:
                pass
        if not (valid_type and valid(input_)):
            if message is not None:
                raise argparse.ArgumentTypeError('{}. Provided "{}".'.format(message, input_))
            else:
                raise argparse.ArgumentTypeError('Input must be one of {}. Provided "{}".'.format(types, input_))
        return input_       
    return _check_valid


def _validate_files_and_directories(arguments):
    with helper.cd(arguments['experiment_dir']):
        files = ['pcs_file', 'instance_file']
        for filename in files:            
            if not helper.isFile(arguments[filename]):
                raise IOError("The {} '{}' could not be found within the current working "
                              "directory '{}' (which is the experiment directory, if one was "
                              "specified)."
                              "".format(filename.replace('_', ' '), arguments[filename], os.getcwd()))
        directories = []
        for directory in directories:            
            if not helper.isDir(arguments[directory]):
                raise IOError("The {} '{}' could not be found within the current working "
                              "directory '{}' (which is the experiment directory, if one was "
                              "specified)."
                              "".format(directory.replace('_', ' '), arguments[directory], os.getcwd()))


def _validate_budget(arguments):
    budgets = ['config_budget', 'wallclock_limit']
    all_default = True
    for budget in budgets:
        all_default = all_default and arguments[budget] == 2147483647
    if all_default:
        raise ValueError('At least one of config_budget and wallclock_limit must be less than '
                         'the maximum integer value (which is their default value).')


def _to_bool(string):
    if string == 'True':
       return True
    elif string == 'False':
        return False
    else:
        raise ValueError("Booleans must be 'True' or 'False'. Provided {}".format(string))

    
def _get_aliases(names):
    aliases = []
    for name in names:
        aliases.append(name)
        if name[:2] == '--':
            # All initial aliases must have dashes and all lower-case letters
            # We replace the dashes with underscores to create an alias
            alias = '--{}'.format(name[2:].replace('-', '_'))
            if alias not in aliases:
                aliases.append(alias)
            # We replace the dashes with cammel case to create an alias
            alias = '--{}{}'.format(name[2:].split('-')[0],
                                    ''.join([token.capitalize() for token in name[2:].split('-')[1:]]))
            if alias not in aliases:
                aliases.append(alias)
            # and we conver the camel case to all lower case letters for yet another alias
            alias = alias.lower()
            if alias not in aliases:
                aliases.append(alias)
    return tuple(aliases)


def _print_argument_documentation():
    """_print_argument_documentation

    Prints out documentation on each of the parameters formated
    to be included in the github readme file, including markdown.
    """
    def _table_row(header, content):
        return '<tr>{}{}</tr>'.format(_table_column(_bold(header)),
                                      _table_column(content))
    def _table_column(content):
        return '<td>{}</td>'.format(content)
    def _bold(header):
        return '<b>{}</b>'.format(header)
    def _list_of_code(aliases):
        return ', '.join([_code(alias.strip()) for alias in aliases])
    def _code(code):
        return '<code>{}</code>'.format(code)
    def _table(description, required, default, aliases):
        return  ('<table>\n{}\n{}\n{}\n</table>\n'
                 ''.format(_table_row('Description', _abreviations_to_italics(description)),
                           _table_row('Required' if required else 'Default',
                                      'Yes' if required else default),
                           _table_row('Aliases', _list_of_code(aliases))))
    def _abreviations_to_italics(content):
        abreviations = ['e.g.', 'i.e.', 'etc.', 'vs.']
        for token in abreviations:
            content = content.replace(token, '<i>{}</i>'.format(token))
        return content

    argument_parser = ArgumentParser()
    defaults, _ = argument_parser.parse_file_arguments(argument_parser.defaults, {})
    for group in argument_parser.groups_in_order:
        print('## {}\n'.format(group))
        print('{}\n'.format(_abreviations_to_italics(argument_parser.group_help[group])))
        arguments = sorted(list(argument_parser.argument_groups[group].keys()))
        for arg in arguments:
            name = _get_name(arg)
            print('### {}\n'.format(name))
            description = argument_parser.argument_groups[group][arg]['help']
            required = name not in defaults
            default = None if required else defaults[name]
            # Handle the one exception to the rule.
            if name == 'scenario_file':
                required = False
                default = None
            # Convert directories to code
            if '_dir' in name:
                default = _code(default)
            aliases = _get_aliases(arg)
            print(_table(description, required, default, aliases))
           
if __name__ == '__main__':
    _print_argument_documentation() 

import sys
import copy
import string
import os
import pickle
from contextlib import contextmanager

import numpy as np

try:
    from ray.tune.sample import uniform, loguniform, randint, choice, sample_from
    ray_available = True
except:
    ray_available = False

import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter as Float
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as Integer


def generateID(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(np.random.choice(list(chars)) for _ in range(size))


def mkdir(dir_):
    #Author: Yasha Pushak
    #Last updated: January 3rd, 2017
    #An alias for makeDir
    makeDir(dir_)


def makeDir(dir_):
    created = []
    for part in dir_.split('/'):
        created.append(part)
        current_dir = '/'.join(created)
        try:
            if not isDir(current_dir):
                os.system('mkdir ' + current_dir)
        except:
            pass


def isDir(dir_):
    #Author: Yasha Pushak
    #last updated: March 21st, 2017
    #Checks if the specified directory exists.
    return os.path.isdir(dir_)


def isFile(filename):
    #Author: Yasha Pushak
    #Last updated: March 21st, 2017
    #CHecks if the specified filename is a file.
    return os.path.isfile(filename)


@contextmanager
def cd(newdir):
    #http://stackoverflow.com/questions/431684/how-do-i-cd-in-python/24176022#24176022
    if len(newdir) > 0:
        prevdir = os.getcwd()
        os.chdir(os.path.expanduser(newdir))
        try:
            yield
        finally:
            os.chdir(prevdir)
    else:
        yield


@contextmanager
def dir_in_path(dir_):
    with cd(dir_):
        sys.path.append('')
        try:
            yield
        finally:
            sys.path = sys.path[:-1]


#Code taken from http://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def saveObj(dir_, obj, name ):
    with open(dir_ + '/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadObj(dir_, name ):
    if isFile(dir_ + '/' + name + '.pkl'):
        filename = dir_ + '/' + name + '.pkl'
    elif isFile(dir_ + '/' + name):
        filename = dir_ + '/' + name
    elif isFile(name + '.pkl'):
        filename = name + '.pkl'
    elif isFile(name):
        filename = name
    else:
        raise IOError("Could not find file '{}' in '{}' or current working directory with or without a "
                      ".pkl extension.".format(name, dir_))
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_obj(*args, **kwargs):
    saveObj(*args, **kwargs)


def load_obj(*args, **kwargs):
    loadObj(*args, **kwargs)


if ray_available:
    def loguniform_int(a, b):
        a = copy.deepcopy(a)
        b = copy.deepcopy(b)
        def _sample():
            return int(loguniform(a, b).sample())
        return _sample
            
    
    def get_configuration_space(config_space):
        configuration = {}
        for hp in config_space.get_hyperparameters():
            name = hp.name
            if isinstance(hp, Integer):
                if hp.log:
                    configuration[name] = sample_from(loguniform_int(hp.lower, hp.upper+1))
                else:
                    configuration[name] = randint(hp.lower, hp.upper+1)
            elif isinstance(hp, Float):
                if hp.log:
                    configuration[name] = loguniform(hp.lower, hp.upper)
                else:
                    configuration[name] = uniform(hp.lower, hp.upper)
            else:
                configuration[name] = choice(hp.choices)
        return configuration


def read_instances(filename):
    if filename != 'None':
        with open(filename, 'r') as f:
            return f.read().split('\n')
    return None
        

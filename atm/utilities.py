from __future__ import absolute_import, unicode_literals

import base64
import hashlib
import json
import logging
import os
import pickle
from builtins import str

import numpy as np

from atm.compat import getargs

logger = logging.getLogger('atm')


def hash_dict(dictionary, ignored_keys=None):
    """
    Hash a python dictionary to a hexadecimal string.
    http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
    """
    dictionary = dict(dictionary)  # copy dictionary
    for key in (ignored_keys or []):
        del dictionary[key]
    return hashlib.md5(repr(sorted(dictionary.items())).encode('utf8')).hexdigest()


def hash_nested_tuple(tup):
    """ Hash a nested tuple to hexadecimal """
    return hashlib.md5(repr(sorted(tup)).encode('utf8')).hexdigest()


def hash_string(s):
    """ Hash a string to hexadecimal """
    return hashlib.md5(str(s).encode('utf8')).hexdigest()


def ensure_directory(directory):
    """ Create directory if it doesn't exist. """
    if not os.path.exists(directory):
        os.makedirs(directory)


def object_to_base_64(obj):
    """ Pickle and base64-encode an object. """
    pickled = pickle.dumps(obj)
    return base64.b64encode(pickled)


def base_64_to_object(b64str):
    """
    Inverse of object_to_base_64.
    Decode base64-encoded string and then unpickle it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


def obj_has_method(obj, method):
    """http://stackoverflow.com/questions/34439/finding-what-methods-an-object-has"""
    return hasattr(obj, method) and callable(getattr(obj, method))


# Converting hyperparameters to and from BTB-compatible formats

def update_params(params, categoricals, constants):
    """
    Update params with categoricals and constants for the fitting proces.

    params: params proposed by the tuner

    Examples of the format for SVM sigmoid hyperpartition:

    categoricals = (('kernel', 'poly'),
                    ('probability', True),
                    ('_scale', True))

    constants = [('cache_size', 15000)]
    """
    for key, value in categoricals + constants:
        params[key] = value

    return params


def get_instance(class_, **kwargs):
    """Instantiate an instance of the given class with unused kwargs

    Args:
        class_ (type): class to instantiate
        **kwargs: keyword arguments to specific selector class

    Returns:
        instance of specific class with the args that accepts.
    """
    init_args = getargs(class_.__init__)
    relevant_kwargs = {
        k: kwargs[k]
        for k in kwargs
        if k in init_args
    }

    return class_(**relevant_kwargs)


def params_to_vectors(params, tunables):
    """
    Converts a list of parameter vectors (with metadata) into a numpy array
    ready for BTB tuning.

    Args:
        params: list of hyperparameter vectors. Each vector is a dict mapping
            the names of parameters to those parameters' values.
        tunables: list of HyperParameter metadata structures describing all
            the optimizable hyperparameters that should be in each vector. e.g.

        tunables = [('C',  HyperParameter(type='float_exp', range=(1e-5, 1e5))),
                    ('degree',  HyperParameter('int', (2, 4))),
                    ('gamma',  HyperParameter('float_exp', (1e-05, 1e5)))]

    Returns:
        vectors: np.array of parameter vectors ready to be optimized by a
            Gaussian Process (or what have you).
            vectors.shape = (len(params), len(tunables))
    """
    # make sure params is iterable
    if not isinstance(params, (list, np.ndarray)):
        params = [params]

    keys = [k[0] for k in tunables]
    vectors = np.zeros((len(params), len(keys)))
    for i, p in enumerate(params):
        for j, k in enumerate(keys):
            vectors[i, j] = p[k]

    return vectors


# Serializing and deserializing data on disk

def make_save_path(dir, classifier, suffix):
    """
    Generate the base save path for a classifier's model and metrics files,
    based on the classifier's dataset name and hyperparameters.
    """
    run_name = "".join([c for c in classifier.datarun.dataset.name
                        if c.isalnum() or c in (' ', '-', '_')]).rstrip()
    params_hash = hash_dict(classifier.hyperparameter_values)[:8]
    filename = "%s-%s.%s" % (run_name, params_hash, suffix)
    return os.path.join(dir, filename)


def save_model(classifier, models_dir, model):
    """
    Save a serialized version of a Model object for a particular classifier.
    The object will be stored at a path generated from the classifier's
    attributes.
    """
    path = make_save_path(models_dir, classifier, 'model')
    logger.info('Saving model in: %s' % path)
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_metrics(classifier, metrics_dir, metrics):
    """
    Save a JSON-serialized version of a set of performance metrics for a
    particular classifier. The metrics will be stored at a path generated from
    the classifier's attributes.
    """
    path = make_save_path(metrics_dir, classifier, 'metric')
    logger.info('Saving metrics in: %s' % path)
    with open(path, 'w') as f:
        json.dump(metrics, f)
    return path


def load_model(classifier, models_dir):
    """ Load the Model object for a particular classifier """
    path = make_save_path(models_dir, classifier, 'model')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_metrics(classifier, metrics_dir):
    """ Load the performance metrics for a particular classifier """
    path = make_save_path(metrics_dir, classifier, 'metric')
    with open(path) as f:
        return json.load(f)

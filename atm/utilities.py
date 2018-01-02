from __future__ import print_function
import json
import pickle
import urllib2
import hashlib
import numpy as np
import os
import base64
import re

from boto.s3.connection import S3Connection, Key

from btb import ParamTypes
from atm.constants import *

# global variable storing this machine's public IP address
# (so we only have to fetch it once)
public_ip = None

# URL which should give us our public-facing IP address
PUBLIC_IP_URL = "http://ip.42.pl/raw"


def hash_dict(dictionary, ignored_keys=None):
    """
    Hash a python dictionary to a hexadecimal string.
    http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
    """
    dictionary = dict(dictionary)  # copy dictionary
    for key in (ignored_keys or []):
        del dictionary[key]
    return hashlib.md5(repr(sorted(dictionary.items()))).hexdigest()


def hash_nested_tuple(tup):
    """ Hash a nested tuple to hexadecimal """
    return hashlib.md5(repr(sorted(tup))).hexdigest()


def hash_string(s):
    """ Hash a string to hexadecimal """
    return hashlib.md5(str(s)).hexdigest()


def ensure_directory(directory):
    """ Create directory if it doesn't exist. """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_public_ip():
    """
    Get the public IP address of this machine. If the request times out,
    return "localhost".
    """
    global public_ip
    if public_ip is None:
        try:
            response = urllib2.urlopen(PUBLIC_IP_URL, timeout=2)
            data = response.read().strip()
            # pull an ip-looking set of numbers from the response
            match = re.search('\d+\.\d+\.\d+\.\d+', data)
            if match:
                public_ip = match.group()
        except Exception as e:  # any exception, doesn't matter what
            print('could not get public IP:', e)
            public_ip = 'localhost'

    return public_ip


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


## Converting hyperparameters to and from BTB-compatible formats

def vector_to_params(vector, tunables, categoricals, constants):
    """
    Converts a numpy vector to a dictionary mapping keys to named parameters.

    vector: single example to convert

    Examples of the format for SVM sigmoid hyperpartition:

    tunables = [('C', HyperParameter(type='float_exp', range=(1e-05, 1e5))),
                ('degree', HyperParameter(type='int', range=(2, 4))),
                ('gamma', HyperParameter(type='float_exp', range=(1e-05, 1e5)))]

    categoricals = (('kernel', 'poly'),
                    ('probability', True),
                    ('_scale', True))

    constants = [('cache_size', 15000)]
    """
    params = {}

    # add the tunables
    for i, elt in enumerate(vector):
        key, struct = tunables[i]
        if struct.type in [ParamTypes.INT, ParamTypes.INT_EXP]:
            params[key] = int(elt)
        elif struct.type in [ParamTypes.FLOAT, ParamTypes.FLOAT_EXP]:
            params[key] = float(elt)
        else:
            raise ValueError('Unknown data type: {}'.format(struct.type))

    # add the fixed categorical settings and fixed constant values
    for key, value in categoricals + constants:
        params[key] = value

    return params


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


## Serializing and deserializing data on disk

def make_save_path(dir, classifier, suffix):
    """
    Generate the base save path for a classifier's model and metrics files,
    based on the classifier's dataset name and hyperparameters.
    """
    run_hash = hash_string(classifier.datarun.dataset.name)
    params_hash = hash_dict(classifier.params)
    filename = "%s-%s-%s.%s" % (run_hash, params_hash,
                                classifier.datarun.description, suffix)
    return os.path.join(dir, filename)


def save_model(classifier, model_dir, model):
    """
    Save a serialized version of a Model object for a particular classifier.
    The object will be stored at a path generated from the classifier's
    attributes.
    """
    path = make_save_path(model_dir, classifier, '.model')
    print('Saving model in: %s' % path)
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_metrics(classifier, metric_dir, metrics):
    """
    Save a JSON-serialized version of a set of performance metrics for a
    particular classifier. The metrics will be stored at a path generated from
    the classifier's attributes.
    """
    path = make_save_path(metric_dir, classifier, '.metric')
    print('Saving metrics in: %s' % path)
    with open(path, 'w') as f:
        json.dump(metrics, f)
    return path


def load_model(classifier, model_dir):
    """ Load the Model object for a particular classifier """
    path = make_save_path(model_dir, classifier, '.model')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_metrics(classifier, metric_dir):
    """ Load the performance metrics for a particular classifier """
    path = make_save_path(metric_dir, classifier, '.metric')
    with open(path) as f:
        return json.load(f)


## Downloading data from the web

def get_local_data_path(data_path):
    """
    given a data path of the form "s3://..." or "http://...", return the local
    path where the file should be stored once it's downloaded.
    """
    if data_path is None:
        return None, None

    m = re.match(S3_PREFIX, data_path)
    if m:
        path = data_path[len(m.group()):].split('/')
        bucket = path.pop(0)
        return os.path.join(DATA_PATH, path[-1]), FileType.S3

    m = re.match(HTTP_PREFIX, data_path)
    if m:
        path = data_path[len(m.group()):].split('/')
        return os.path.join(DATA_PATH, path[-1]), FileType.HTTP

    return data_path, FileType.LOCAL


def download_file_s3(aws_path, aws_config, local_folder=DATA_PATH):
    """ Download a file from an S3 bucket and save it in the local folder. """
    # remove the prefix and extract the S3 bucket, folder, and file name
    m = re.match(S3_PREFIX, aws_path)
    split = aws_path[len(m.group()):].split('/')
    s3_bucket = split.pop(0)
    s3_folder = '/'.join(split[:-1])
    keyname = split[-1]

    # create the local folder if necessary
    if local_folder is not None:
        ensure_directory(local_folder)
        path = os.path.join(local_folder, keyname)
    else:
        path = keyname

    if os.path.isfile(path):
        print('file %s already exists!' % path)
        return path

    conn = S3Connection(aws_config.access_key, aws_config.secret_key)
    bucket = conn.get_bucket(s3_bucket)

    if s3_folder:
        aws_keyname = os.path.join(s3_folder, keyname)
    else:
        aws_keyname = keyname

    print('downloading data from S3...')
    s3key = Key(bucket)
    s3key.key = aws_keyname
    s3key.get_contents_to_filename(path)

    return path


def download_file_http(url, local_folder=DATA_PATH):
    """ Download a file from a public URL and save it locally. """
    filename = url.split('/')[-1]
    if local_folder is not None:
        ensure_directory(local_folder)
        path = os.path.join(local_folder, filename)
    else:
        path = filename

    if os.path.isfile(path):
        print('file %s already exists!' % path)
        return path

    print('downloading data from %s...' % url)
    f = urllib2.urlopen(url)
    data = f.read()
    with open(path, "wb") as outfile:
        outfile.write(data)

    return path


def download_data(train_path, test_path=None, aws_config=None):
    """
    Download a set of train/test data from AWS (if necessary) and return the
    path to the local data.
    """
    local_train_path, train_type = get_local_data_path(train_path)
    local_test_path, test_type = get_local_data_path(test_path)

    # if the data are not present locally, try to download them from the
    # internet (either an S3 bucket or a http connection)
    if not os.path.isfile(local_train_path):
        if train_type == FileType.HTTP:
            assert (download_file_http(train_path) == local_train_path)
        elif train_type == FileType.S3:
            assert (download_file_s3(train_path, aws_config=aws_config) ==
                    local_train_path)

    if local_test_path and not os.path.isfile(local_test_path):
        if test_type == FileType.HTTP:
            assert (download_file_http(test_path) == local_test_path)
        elif test_type == FileType.S3:
            assert (download_file_s3(test_path, aws_config=aws_config) ==
                    local_test_path)

    return local_train_path, local_test_path

import pickle
import urllib2
import hashlib
import numpy as np
import os
import base64
import re

from boto.s3.connection import S3Connection, Key

from btb import ParamTypes

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
            print 'could not get public IP:', e
            public_ip = 'localhost'

    return public_ip


def object_to_base_64(obj):
    """ Pickle and base64-encode an object. """
    pickled = pickle.dumps(obj)
    return base64.b64encode(pickled)


def base_64_to_object(b64str):
    """
    Inverse of ObjectToBase64.
    Decode base64-encoded string and then de-pickle it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


def obj_has_method(obj, method):
    """http://stackoverflow.com/questions/34439/finding-what-methods-an-object-has"""
    return hasattr(obj, method) and callable(getattr(obj, method))


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


def make_model_path(model_dir, params_hash, run_hash, desc):
    filename = "%s-%s-%s.model" % (run_hash, params_hash, desc)
    return os.path.join(model_dir, filename)


def make_metric_path(metric_dir, params_hash, run_hash, desc):
    filename = "%s-%s-%s.metric" % (run_hash, params_hash, desc)
    return os.path.join(metric_dir, filename)


def save_metric(metric_path, object):
    with open(metric_path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_atm_csv(filepath):
    """
    Read a csv and return a numpy array.
    This works from the assumption the data has been preprocessed by atm:
    no headers, numerical data only
    """
    num_cols = len(open(filepath).readline().split(','))
    with open(filepath) as f:
        for i, _ in enumerate(f):
            pass
    num_rows = i + 1

    data = np.zeros((num_rows, num_cols))

    with open(filepath) as f:
        for i, line in enumerate(f):
            for j, cell in enumerate(line.split(',')):
                data[i, j] = float(cell)

    return data


def download_file_s3(keyname, aws_key, aws_secret, s3_bucket,
                     s3_folder=None, local_folder=None):
    """ Download a file from an S3 bucket and save it at keyname.  """
    if local_folder:
        path = os.path.join(local_folder, keyname)
    else:
        path = keyname

    if os.path.isfile(path):
        print 'file %s already exists!' % path
        return path

    conn = S3Connection(aws_key, aws_secret)
    bucket = conn.get_bucket(s3_bucket)

    if s3_folder:
        aws_keyname = os.path.join(s3_folder, keyname)
    else:
        aws_keyname = keyname

    print 'downloading data from S3...'
    s3key = Key(bucket)
    s3key.key = aws_keyname
    s3key.get_contents_to_filename(path)

    return path

import pickle
import urllib2
import hashlib
import numpy as np
import os
import base64
import re
from boto.s3.connection import S3Connection, Key
from atm.config import Config

# global variable storing this machine's public IP address
# (so we only have to fetch it once)
public_ip = None
PUBLIC_IP_URL = "http://ip.42.pl/raw"


def hash_dict(dictionary, ignored_keys=[]):
    """
    http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
    """
    dictionary = dict(dictionary)  # copy dictionary
    for key in ignored_keys:
        del dictionary[key]
    return hashlib.md5(repr(sorted(dictionary.items()))).hexdigest()


def hash_nested_tuple(tup):
    return hashlib.md5(repr(sorted(tup))).hexdigest()


def hash_string(s):
    return hashlib.md5(str(s)).hexdigest()


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_public_ip():
    global public_ip
    if public_ip is None:
        try:
            response = urllib2.urlopen(PUBLIC_IP_URL).read().strip()
            match = re.search('\d+\.\d+\.\d+\.\d+', response)
            if match:
                public_ip = match.group()
        except Exception as e:  # any exception, doesn't matter
            print 'could not get public IP:', e
            public_ip = 'localhost'

    return public_ip


def object_to_base_64(obj):
    """
        Takes object in memory, then pickles and
        base64 encodes it.
    """
    pickled = pickle.dumps(obj)
    return base64.b64encode(pickled)


def base_64_to_object(b64str):
    """
        Inverse of ObjectToBase64.

        Decodes base64 encoded string and
        then de-pickles it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)


def obj_has_method(obj, method):
    """http://stackoverflow.com/questions/34439/finding-what-methods-an-object-has"""
    return hasattr(obj, method) and callable(getattr(obj, method))


def vector_to_params(vector, optimizables, frozens, constants):
    """
    Converts a numpy vector to a dictionary mapping keys to named parameters.

    `vector` is single example to convert
    `optimizables` keys back from vector format to dictionaries.

    Examples of the format for SVM sigmoid frozen set below:

        optimizables = [
            ('C', 		KeyStruct(range=(1e-05, 100000), 	type='FLOAT_EXP', 	is_categorical=False)),
            ('degree', 	KeyStruct(range=(2, 4), 			type='INT', 		is_categorical=False)),
            ('coef0', 	KeyStruct(range=(0, 1), 			type='INT', 		is_categorical=False)),
            ('gamma', 	KeyStruct(range=(1e-05, 100000),	type='FLOAT_EXP', 	is_categorical=False))
        ]

        frozens = (
            ('kernel', 'poly'), ('probability', True),
            ('_scale', True), ('shrinking', True),
            ('class_weight', 'auto')
        )

        constants = [
            ('cache_size', KeyStruct(range=(15000, 15000), type='INT', is_categorical=False))
        ]

    """
    params = {}

    # add the optimizables
    for i, elt in enumerate(vector):
        key, struct = optimizables[i]
        if struct.type == 'INT':
            params[key] = int(elt)
        elif struct.type == 'INT_EXP':
            params[key] = int(elt)
        elif struct.type == 'FLOAT':
            params[key] = float(elt)
        elif struct.type == 'FLOAT_EXP':
            params[key] = float(elt)
        elif struct.type == 'BOOL':
            params[key] = bool(elt)
        else:
            raise ValueError('Unknown data type: {}'.format(struct.type))

    # add the frozen categorical settings
    for key, value in frozens:
        params[key] = value

    # and finally the constant values
    for constant_key, struct in constants:
        params[constant_key] = struct.range[0]

    return params


def params_to_vectors(params, optimizables):
    """
    Converts a list of parameter vectors (with metadata) into a numpy array
    ready for BTB tuning.

    Args:
        params: list of hyperparameter vectors. Each vector is a dict mapping
            the names of parameters to those parameters' values.
        optimizables: list of HyperParameter metadata structures describing all
            the optimizable hyperparameters that should be in each vector. e.g.

        optimizables = [
            ('C',  HyperParameter(range=(1e-5, 1e5), type='FLOAT_EXP', is_categorical=False)),
            ('degree',  HyperParameter((2, 4), 'INT', False)),
            ('coef0',  HyperParameter((0, 1), 'INT', False)),
            ('gamma',  HyperParameter((1e-05, 100000), 'FLOAT_EXP', False))
        ]

    Returns:
        vectors: np.array of parameter vectors ready to be optimized by a
            Gaussian Process (or what have you).
            vectors.shape = (len(params), len(optimizables))
    """
    # make sure params is iterable
    if not isinstance(params, (list, np.ndarray)):
        params = [params]

    keys = [k[0] for k in optimizables]
    vectors = np.zeros((len(params), len(keys)))
    for i, p in enumerate(params):
        for j, k in enumerate(keys):
            vectors[i, j] = p[k]
    return vectors


def make_model_path(model_dir, params_hash, run_hash, desc):
    return os.path.join(model_dir, "%s-%s-%s.model" % (run_hash, params_hash, desc))


def make_metric_path(model_dir, params_hash, run_hash, desc):
    return os.path.join(model_dir, "%s-%s-%s.metric" % (run_hash, params_hash, desc))


def save_metric(metric_path, object):
    with open(metric_path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def download_file_s3(config, keyname):
    aws_key = config.get(Config.AWS, Config.AWS_ACCESS_KEY)
    aws_secret = config.get(Config.AWS, Config.AWS_SECRET_KEY)
    s3_bucket = config.get(Config.AWS, Config.AWS_S3_BUCKET)
    s3_folder = config.get(Config.AWS, Config.AWS_S3_FOLDER).strip()

    print 'getting S3 connection...'
    conn = S3Connection(aws_key, aws_secret)
    bucket = conn.get_bucket(s3_bucket)

    if s3_folder:
        aws_keyname = os.path.join(s3_folder, keyname)
    else:
        aws_keyname = keyname

    s3key = Key(bucket)
    s3key.key = aws_keyname
    s3key.get_contents_to_filename(keyname)

    return keyname

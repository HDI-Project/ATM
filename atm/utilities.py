import pickle
import urllib2
import hashlib
import numpy as np
import os
import base64

PUBLIC_IP_URL = "http://ifconfig.me/ip"  # "http://ipecho.net/plain"


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


public_ip = None
def get_public_ip():
    global public_ip
    if public_ip is None:
        try:
            response = urllib2.urlopen(PUBLIC_IP_URL).read().strip()
            match = re.search('\d+\.\d+\.\d+\.\d+', response)
            if match:
                public_ip = match.group()
        except Exception:  # any exception, doesn't matter
            pass

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
    params is a list of {'key' -> value}

    Example of optimizables below:

    optimizables = [
        ('C', 		KeyStruct(range=(1e-05, 100000), 	type='FLOAT_EXP', 	is_categorical=False)),
        ('degree', 	KeyStruct(range=(2, 4), 			type='INT', 		is_categorical=False)),
        ('coef0', 	KeyStruct(range=(0, 1), 			type='INT', 		is_categorical=False)),
        ('gamma', 	KeyStruct(range=(1e-05, 100000),	type='FLOAT_EXP', 	is_categorical=False))
    ]

    Creates vectors ready to be optimized by a Gaussian Process.
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

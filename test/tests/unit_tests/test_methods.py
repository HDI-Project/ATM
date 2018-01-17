#!/usr/bin/python2.7
import pytest
import json
from collections import defaultdict
from os.path import join

from atm.config import RunConfig
from atm.database import Database
from atm.enter_data import create_dataset, create_datarun
from atm.utilities import download_file_s3
from atm.worker import work

from utilities import work_parallel


def test_enumerate():
    js = {'name': 'test', 'class': 'test'}
    js['hyperparameters'] = {
        'a': {'type': 'int_cat', 'range': [0, 3]},
        'b': {'type': 'int', 'range': [0, 3]},
        'c': {'type': 'bool', 'range': [True, False]},
        'd': {'type': 'string', 'range': ['x', 'y']},
        'e': {'type': 'float_cat', 'range': [-0.5, 0.5, 1.0]},
        'f': {'type': 'float', 'range': [0.5]},
    }
    js['root_parameters'] = ['a', 'f']
    js['conditions'] = {
        'a': {'1': 'b', '2': 'c'},
        'c': {'True': 'd', 'False': 'e'},
    }

    config_path = '/tmp/method.json'
    with open(config_path, 'w') as f:
        json.dump(js, f)

    hp = Method(config_path).get_hyperpartitions()

    assert len(hp) == 8
    #assert all('a' in zip(*p.categoricals)[0] for p in hp)
    #assert all(('f', 0.5) in p.constants for p in hp)
    #assert len(['b' in zip(*p.categoricals)[0] for p in hp])

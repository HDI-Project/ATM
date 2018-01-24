#!/usr/bin/python2.7
import pytest
import json

from atm.method import Method


def test_enumerate():
    js = {'name': 'test', 'class': 'test'}
    js['hyperparameters'] = {
        'a': {'type': 'int_cat', 'values': [0, 3]},
        'b': {'type': 'int', 'range': [0, 3]},
        'c': {'type': 'bool', 'values': [True, False]},
        'd': {'type': 'string', 'values': ['x', 'y']},
        'e': {'type': 'float_cat', 'values': [-0.5, 0.5, 1.0]},
        'f': {'type': 'float', 'range': [0.5]},
        'g': {'type': 'list',
              'list_length': [1, 2, 3],
              'element': {'type': 'int_exp', 'range': [1e-3, 1e3]}}
    }
    js['root_hyperparameters'] = ['a', 'f']
    js['conditional_hyperparameters'] = {
        'a': {'0': ['b'], '3': ['c']},
        'c': {'True': ['d'], 'False': ['e', 'g']},
    }

    config_path = '/tmp/method.json'
    with open(config_path, 'w') as f:
        json.dump(js, f)

    hps = Method(config_path).get_hyperpartitions()

    assert len(hps) == 12
    assert all('a' in zip(*hp.categoricals)[0] for hp in hps)
    assert all(('f', 0.5) in hp.constants for hp in hps)
    assert len([hp for hp in hps if hp.tunables
                and 'b' in zip(*hp.tunables)[0]]) == 1

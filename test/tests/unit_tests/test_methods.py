#!/usr/bin/python2.7
import pytest
import json

from atm.method import Method


def test_enumerate():
    js = {'name': 'test', 'class': 'test'}
    js['parameters'] = {
        'a': {'type': 'int_cat', 'range': [0, 3]},
        'b': {'type': 'int', 'range': [0, 3]},
        'c': {'type': 'bool', 'range': [True, False]},
        'd': {'type': 'string', 'range': ['x', 'y']},
        'e': {'type': 'float_cat', 'range': [-0.5, 0.5, 1.0]},
        'f': {'type': 'float', 'range': [0.5]},
    }
    js['root_parameters'] = ['a', 'f']
    js['conditions'] = {
        'a': {'0': 'b', '3': 'c'},
        'c': {'True': 'd', 'False': 'e'},
    }

    config_path = '/tmp/method.json'
    with open(config_path, 'w') as f:
        json.dump(js, f)

    hp = Method(config_path).get_hyperpartitions()

    assert len(hp) == 6
    assert all('a' in zip(*p.categoricals)[0] for p in hp)
    assert all(('f', 0.5) in p.constants for p in hp)
    assert len(['b' in zip(*p.categoricals)[0] for p in hp])

# -*- coding: utf-8 -*-

"""Backwards compaitibility module.

This module contains functions to ensure compatibility with
both Python 2 and 3
"""
import inspect

from six import PY2


def getargs(function):
    """Get the function arguments using inspect."""
    if PY2:
        return inspect.getargspec(function).args
    else:
        return inspect.getfullargspec(function).args

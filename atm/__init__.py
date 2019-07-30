"""Auto Tune Models
A multi-user, multi-data AutoML framework.
"""
from __future__ import absolute_import, unicode_literals

import os

from atm.classifier import Model
from atm.core import ATM

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.3-dev'

# this defines which modules will be imported by "from atm import *"
__all__ = ['ATM', 'Model', 'config', 'constants', 'data', 'database',
           'method', 'metrics', 'models', 'utilities', 'worker']

# Get the path of the project root, so that the rest of the project can
# reference files relative to there.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

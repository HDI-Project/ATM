"""Auto Tune Models
A multi-user, multi-data AutoML framework.
"""
from __future__ import absolute_import, unicode_literals

import logging
import os

# Get the path of the project root, so that the rest of the project can
# reference files relative to there.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# this defines which modules will be imported by "from atm import *"
__all__ = ['config', 'classifier', 'constants', 'database', 'enter_data',
           'method', 'metrics', 'models', 'utilities', 'worker']

# by default, nothing should be logged
logger = logging.getLogger('atm')
logger.addHandler(logging.NullHandler())

"""Auto Tune Models
A multi-user, multi-data AutoML framework.
"""
from __future__ import absolute_import
import os

# Get the path of the project root, so that the rest of the project can
# reference files relative to there.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

__all__ = ['config', 'constants', 'database', 'enter_data', 'method', 'metrics',
           'model', 'utilities', 'worker']

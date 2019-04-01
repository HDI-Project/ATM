import inspect

from six import PY2


def getargs(func):
    if PY2:
        return inspect.getargspec(func).args
    else:
        return inspect.getfullargspec(func).args

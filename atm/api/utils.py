import logging
import os
import traceback

import flask

LOGGER = logging.getLogger(__name__)


def make_absolute(url):
    if str(url).startswith('sqlite:///'):
        url = 'sqlite:///' + os.path.abspath(url.database)

    return url


def abort(code, message=None, error=None):
    if error is not None:
        error = traceback.format_exception_only(type(error), error)[0].strip()

    response = flask.jsonify({
        'status': code,
        'error': error,
        'message': message
    })
    response.status_code = code
    flask.abort(response)


def auto_abort(exceptions):
    def outer(function):
        def inner(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except exceptions as ex:
                abort(400, error=ex)
            except Exception as ex:
                LOGGER.exception('Uncontrolled Exception Caught')
                abort(500, error=ex)

        return inner

    return outer

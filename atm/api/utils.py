import os
import traceback

import flask


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

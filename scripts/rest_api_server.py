# coding: utf-8

import os
import copy
import uuid
import decimal
import datetime
import argparse
import simplejson as json
from sqlalchemy import inspect
from subprocess import Popen, PIPE
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from atm.database import Database
from atm.enter_data import enter_data
from atm.config import (add_arguments_aws_s3, add_arguments_sql,
                        add_arguments_datarun, add_arguments_logging,
                        load_config, initialize_logging)


class JSONEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that knows how to encode date/time, decimal types, and UUIDs.
    See: https://stackoverflow.com/questions/11875770/how-to-overcome-datetime-datetime-not-json-serializable
    """
    def default(self, o):
        # See "Date Time String Format" in the ECMA-262 specification.
        if isinstance(o, datetime.datetime):
            r = o.isoformat()
            if o.microsecond:
                r = r[:23] + r[26:]
            if r.endswith('+00:00'):
                r = r[:-6] + 'Z'
            return r
        elif isinstance(o, datetime.date):
            return o.isoformat()
        elif isinstance(o, datetime.time):
            if o.utcoffset() is not None:
                raise ValueError("JSON can't represent timezone-aware times.")
            r = o.isoformat()
            if o.microsecond:
                r = r[:12]
            return r
        elif isinstance(o, (decimal.Decimal, uuid.UUID)):
            return str(o)
        else:
            return super(JSONEncoder, self).default(o)


class ApiError(Exception):
    """
    API error handler Exception
    See: http://flask.pocoo.org/docs/0.12/patterns/apierrors/
    """
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def table_fetcher(table):
    """
    Creates a generic controller function to view the full contents of a table.
    """

    def inner():
        result = db.engine.execute(''.join(['SELECT * FROM ', table]))
        return json.dumps([dict(row) for row in result])

    return inner


def entity_fetcher(entity, field, one=False):
    """
    Creates a generic controller function to filter the entity by the value of one field.

    Uses simplejson (aliased to json) to parse Decimals and the custom JSONEncoder to parse
    datetime fields.
    """

    def inner(**args):
        value = args[field]
        kwargs = {field: value}

        try:
            if one:
                result = session.query(entity).filter_by(**kwargs).one()
                return json.dumps((object_as_dict(result)), cls=JSONEncoder)
            else:
                result = session.query(entity).filter_by(**kwargs).all()
                return json.dumps([object_as_dict(item) for item in result], cls=JSONEncoder)

        except Exception:
            raise ApiError('Not found', status_code=404, payload={})

    return inner


def allowed_file(filename):
    """
    Checks if filename ends with an allowed file extension.
    See: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def post_enter_data():
    """
    Receives and saves a CSV file, after which it executes the enter_data function.
    See: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
    """
    if 'file' not in request.files:
        raise ApiError('No file part', status_code=400)

    file = request.files['file']

    # if user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        raise ApiError('Empty file part', status_code=400)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        rel_filepath = os.path.join(UPLOAD_FOLDER, filename)
        abs_filepath = os.path.abspath(rel_filepath)
        file.save(rel_filepath)

        # we need to set a customized train_path but without modifying the
        # global run_conf object, so we deepcopy the run_conf object
        upload_run_conf = copy.deepcopy(run_conf)
        upload_run_conf.train_path = abs_filepath

        enter_data(sql_conf, upload_run_conf, aws_conf, _args.run_per_partition)

        return json.dumps({'success': True})


def execute_in_virtualenv(virtualenv_name, script):
    """
    Executes a Python script inside a virtualenv.
    See: https://gist.github.com/turicas/2897697
    General idea:
    /bin/bash -c "source venv/bin/activate && python /home/jose/code/python/ATM/worker.py"
    """
    path = ''.join([os.path.dirname(os.path.abspath(__file__)), script])
    command = ''.join(['/bin/bash -c "source venv/bin/activate && python ', path, '"'])
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    return process


def dispatch_worker():
    """
    Executes the worker.py script inside a virtualenv and returns stdout and stderr
    as response.

    Note: It currently only works if rest_api_server.py file is in the same
    directory as the worker.py script.
    """
    process = execute_in_virtualenv('venv', '/worker.py')
    stdout, stderr = process.communicate()

    return jsonify({
        'stdout': stdout,
        'stderr': stderr
    })


if __name__ == '__main__':
    # ATM flags
    parser = argparse.ArgumentParser()
    add_arguments_aws_s3(parser)
    add_arguments_sql(parser)
    add_arguments_datarun(parser)
    add_arguments_logging(parser)
    parser.add_argument('--run-per-partition', default=False, action='store_true',
                        help='if set, generate a new datarun for each hyperpartition')

    # API flags
    parser.add_argument('--host', default='localhost', help='Port in which to run the API')
    parser.add_argument('--port', default=5000, help='Port in which to run the API')
    parser.add_argument('--debug', default=False, help='If true, run Flask in debug mode')
    _args = parser.parse_args()

    # global configuration objects
    config = load_config(sql_path=_args.sql_config,
                         run_path=_args.run_config,
                         aws_path=_args.aws_config,
                         log_path=_args.log_config,
                         **vars(_args))

    sql_conf, run_conf, aws_conf, log_conf = config

    # global database object
    db = Database(sql_conf.dialect, sql_conf.database, sql_conf.username,
                  sql_conf.password, sql_conf.host, sql_conf.port,
                  sql_conf.query)

    # flask app
    virtualenv = 'venv'
    app = Flask(__name__)
    app.config.from_object(__name__)
    session = db.get_session()

    UPLOAD_FOLDER = 'atm/data'
    ALLOWED_EXTENSIONS = set(['csv'])


    @app.errorhandler(ApiError)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response


    def object_as_dict(obj):
        return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}


    # routes to find all records
    app.add_url_rule('/datasets', 'all_datasets', table_fetcher('datasets'), methods=['GET'])
    app.add_url_rule('/dataruns', 'all_dataruns', table_fetcher('dataruns'), methods=['GET'])
    app.add_url_rule('/hyperpartitions', 'all_hyperpartitions', table_fetcher('hyperpartitions'), methods=['GET'])
    app.add_url_rule('/classifiers', 'all_classifiers', table_fetcher('classifiers'), methods=['GET'])

    # routes to find entity by it's own id
    app.add_url_rule('/dataruns/<int:id>', 'datarun_by_id',
                     entity_fetcher(db.Datarun, 'id', one=True), methods=['GET'])
    app.add_url_rule('/datasets/<int:id>', 'dataset_by_id',
                     entity_fetcher(db.Dataset, 'id', one=True), methods=['GET'])
    app.add_url_rule('/classifiers/<int:id>', 'classifier_by_id',
                     entity_fetcher(db.Classifier, 'id', one=True), methods=['GET'])
    app.add_url_rule('/hyperpartitions/<int:id>', 'hyperpartition_by_id',
                     entity_fetcher(db.Hyperpartition, 'id', one=True), methods=['GET'])

    # routes to find entities associated with another entity
    app.add_url_rule('/dataruns/dataset/<int:dataset_id>', 'datarun_by_dataset_id',
                     entity_fetcher(db.Datarun, 'dataset_id'), methods=['GET'])
    app.add_url_rule('/hyperpartitions/datarun/<int:datarun_id>', 'hyperpartition_by_datarun_id',
                     entity_fetcher(db.Hyperpartition, 'datarun_id'), methods=['GET'])
    app.add_url_rule('/classifiers/datarun/<int:datarun_id>', 'classifier_by_datarun_id',
                     entity_fetcher(db.Classifier, 'datarun_id'), methods=['GET'])
    app.add_url_rule('/classifiers/hyperpartition/<int:hyperpartition_id>', 'classifier_by_hyperpartition_id',
                     entity_fetcher(db.Classifier, 'hyperpartition_id'), methods=['GET'])

    # route to post a new CSV file and create a datarun with enter_data
    app.add_url_rule('/enter_data', 'enter_data', post_enter_data, methods=['POST'])

    # route to activate a single worker
    app.add_url_rule('/simple_worker', 'simple_worker', dispatch_worker, methods=['GET'])

    app.run(
        debug=_args.debug,
        host=_args.host,
        port=int(_args.port)
    )

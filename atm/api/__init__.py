import os

from flask import Flask, abort, jsonify, redirect, request
from flask_restless_swagger import SwagAPIManager as APIManager
from flask_sqlalchemy import SQLAlchemy

from atm.api.preprocessors import DATASET_PREPROCESSORS
from atm.config import RunConfig


def make_absolute(url):
    if str(url).startswith('sqlite:///'):
        url = 'sqlite:///' + os.path.abspath(url.database)

    return url


def create_app(atm):
    db = atm.db
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = make_absolute(db.engine.url)

    # Create the Flask-Restless API manager.
    manager = APIManager(app, flask_sqlalchemy_db=SQLAlchemy(app))

    @app.route('/api/run', methods=['POST'])
    def atm_run():
        if not request.json:
            abort(400)

        data = request.json
        run_per_partition = data.get('run_per_partition', False)
        run_conf = RunConfig(data)

        dataruns = atm.create_dataruns(run_conf, run_per_partition)

        response = {
            'status': 'OK',
            'datarun_ids': [datarun.id for datarun in dataruns]
        }

        return jsonify(response)

    @app.route('/')
    def swagger():
        return redirect('/static/swagger/swagger-ui/index.html')

    # Create API endpoints, which will be available at /api/<tablename> by
    # default. Allowed HTTP methods can be specified as well.
    manager.create_api(db.Dataset, methods=['GET', 'POST'], preprocessors=DATASET_PREPROCESSORS)
    manager.create_api(db.Datarun, methods=['GET'])
    manager.create_api(db.Hyperpartition, methods=['GET'])
    manager.create_api(db.Classifier, methods=['GET'])

    return app

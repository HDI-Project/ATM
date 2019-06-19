from flask import Flask, jsonify, redirect, request
from flask_restless_swagger import SwagAPIManager as APIManager
from flask_sqlalchemy import SQLAlchemy

from atm.api.utils import auto_abort, make_absolute
from atm.config import RunConfig


def create_app(atm, debug=False):
    db = atm.db
    app = Flask(__name__)
    app.config['DEBUG'] = debug
    app.config['SQLALCHEMY_DATABASE_URI'] = make_absolute(db.engine.url)

    # Create the Flask-Restless API manager.
    manager = APIManager(app, flask_sqlalchemy_db=SQLAlchemy(app))

    # Allow the CORS header
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    @app.route('/api/run', methods=['POST'])
    @auto_abort((KeyError, ValueError))
    def atm_run():
        data = request.json
        run_conf = RunConfig(data)

        dataruns = atm.create_dataruns(run_conf)

        response = {
            'status': 200,
            'datarun_ids': [datarun.id for datarun in dataruns]
        }

        return jsonify(response)

    @app.route('/')
    def swagger():
        return redirect('/static/swagger/swagger-ui/index.html')

    # Create API endpoints, which will be available at /api/<tablename> by
    # default. Allowed HTTP methods can be specified as well.
    manager.create_api(db.Dataset, methods=['GET', 'POST'])
    manager.create_api(db.Datarun, methods=['GET'])
    manager.create_api(db.Hyperpartition, methods=['GET'])
    manager.create_api(db.Classifier, methods=['GET'])

    return app

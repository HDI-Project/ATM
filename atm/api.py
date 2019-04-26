import os

from flask import Flask, redirect
from flask_restless_swagger import SwagAPIManager as APIManager
from flask_sqlalchemy import SQLAlchemy


def make_absolute(url):
    if str(url).startswith('sqlite:///'):
        url = 'sqlite:///' + os.path.abspath(url.database)

    return url


def create_app(db, debug=False):
    app = Flask(__name__)
    app.config['DEBUG'] = debug
    app.config['SQLALCHEMY_DATABASE_URI'] = make_absolute(db.engine.url)

    # Create the Flask-Restless API manager.
    manager = APIManager(app, flask_sqlalchemy_db=SQLAlchemy(app))

    # Create API endpoints, which will be available at /api/<tablename> by
    # default. Allowed HTTP methods can be specified as well.

    @app.route('/')
    def swagger():
        return redirect('/static/swagger/swagger-ui/index.html')

    manager.create_api(db.Dataset, methods=['GET'])
    manager.create_api(db.Datarun, methods=['GET'])
    manager.create_api(db.Hyperpartition, methods=['GET'])
    manager.create_api(db.Classifier, methods=['GET'])

    return app

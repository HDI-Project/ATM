import os

from flask import Flask, redirect
from flask_restless_swagger import SwagAPIManager as APIManager
from flask_sqlalchemy import SQLAlchemy


def make_absolute(url):
    if str(url).startswith('sqlite:///'):
        url = 'sqlite:///' + os.path.abspath(url.database)

    return url


def create_app(atm):
    app = Flask(__name__)
    app.config['DEBUG'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = make_absolute(atm.db.engine.url)
    db = SQLAlchemy(app)

    # Create the Flask-Restless API manager.
    manager = APIManager(app, flask_sqlalchemy_db=db)

    # Create API endpoints, which will be available at /api/<tablename> by
    # default. Allowed HTTP methods can be specified as well.

    @app.route('/')
    def swagger():
        return redirect('/static/swagger/swagger-ui/index.html')

    manager.create_api(atm.db.Dataset, methods=['GET'])
    manager.create_api(atm.db.Datarun, methods=['GET'])
    manager.create_api(atm.db.Hyperpartition, methods=['GET'])
    manager.create_api(atm.db.Classifier, methods=['GET'])

    return app

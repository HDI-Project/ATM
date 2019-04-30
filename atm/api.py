import os
from past.utils import old_div

from flask import Flask, abort, redirect, request
from flask_restless_swagger import SwagAPIManager as APIManager
from flask_sqlalchemy import SQLAlchemy

from atm.encoder import MetaData

DATASET_KEYS = ['name', 'description', 'train_path', 'class_column']


def make_absolute(url):
    if str(url).startswith('sqlite:///'):
        url = 'sqlite:///' + os.path.abspath(url.database)

    return url


def dataset_preprocessor(data):
    """Preprocess the post data."""
    if all(key in data for key in DATASET_KEYS):
        meta = MetaData(
            data['class_column'],
            data['train_path'],
            data.get('test_path')
        )

        data['n_examples'] = meta.n_examples
        data['k_classes'] = meta.k_classes
        data['d_features'] = meta.d_features
        data['majority'] = meta.majority
        data['size_kb'] = old_div(meta.size, 1000)

    else:
        abort(400)


DATASET_PREPROCESSOR = {'POST_RESOURCE': [dataset_preprocessor]}


def create_app(atm):
    app = Flask(__name__)
    app.config['DEBUG'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = make_absolute(atm.db.engine.url)
    db = SQLAlchemy(app)

    # Create the Flask-Restless API manager.
    manager = APIManager(app, flask_sqlalchemy_db=db)

    # Create API endpoints, which will be available at /api/<tablename> by
    # default. Allowed HTTP methods can be specified as well.

    @app.route('/api/search', methods=['POST'])
    def create_datarun():
        if not request.json:
            abort(400)
        return

    @app.route('/')
    def swagger():
        return redirect('/static/swagger/swagger-ui/index.html')

    manager.create_api(atm.db.Dataset, methods=['GET', 'POST'], preprocessors=DATASET_PREPROCESSOR)
    manager.create_api(atm.db.Datarun, methods=['GET'])
    manager.create_api(atm.db.Hyperpartition, methods=['GET'])
    manager.create_api(atm.db.Classifier, methods=['GET'])

    return app

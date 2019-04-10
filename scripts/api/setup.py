import os

from flask import Flask
from flask_restplus import Api
from werkzeug.contrib.fixers import ProxyFix

from atm.config import load_config
from atm.database import Database


def set_up_flask():
    app = Flask(__name__)

    app.wsgi_app = ProxyFix(app.wsgi_app)

    api = Api(app, version='0.1', title='ATM API',
              description='A RESTful API for Auto Tuning Models')

    ns = api.namespace('api', description='ATM API operations')

    return (app, api, ns)

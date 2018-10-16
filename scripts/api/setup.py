import os

from flask import Flask
from flask_restplus import Api
from werkzeug.contrib.fixers import ProxyFix

from atm.config import load_config
from atm.database import Database


def set_up_db():
    sql_config_path = os.path.join('config', 'sql.yaml')
    sql_conf = load_config(sql_path=sql_config_path)[0]

    # YOU NEED TO redo SQL_CONF path, or get database to accept somethign in a
    # higher directory

    sql_conf.database = os.path.join(sql_conf.database)

    db = Database(sql_conf.dialect, sql_conf.database, sql_conf.username,
                  sql_conf.password, sql_conf.host, sql_conf.port,
                  sql_conf.query)

    return db


def set_up_flask():
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app)
    api = Api(app, version='0.1', title='ATM API',
              description='A RESTful API for Auto Tuning Models')
    ns = api.namespace('api', description='ATM API operations')

    return (app, api, ns)

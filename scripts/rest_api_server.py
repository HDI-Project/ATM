import datetime
import decimal
import os
import simplejson as json
import uuid
import operator

from flask import Flask
from flask_restplus import Api, Resource, reqparse
from sqlalchemy import inspect
from werkzeug.contrib.fixers import ProxyFix

from atm import PROJECT_ROOT
from atm.database import Database
from atm.config import load_config


def set_up_db():
    sql_config_path = os.path.join(PROJECT_ROOT, '..', 'config', 'sql.yaml')
    sql_conf = load_config(sql_path=sql_config_path)[0]

    # YOU NEED TO redo SQL_CONF path, or get database to accept somethign in a
    # higher directory

    sql_conf.database = os.path.join('..', sql_conf.database)

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


def object_as_dict(obj):
        return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}  # noqa


class JSONEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that knows how to encode date/time, decimal types, and
    UUIDs.
    See: https://stackoverflow.com/questions/11875770/how-to-overcome-datetime-datetime-not-json-serializable # noqa
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


def encode_entity(entity=[]):
    """
    Creates a generic controller function to filter the entity by the value of
    one field.

    Uses simplejson (aliased to json) to parse Decimals and the custom
    JSONEncoder to parse datetime fields.
    """
    return json.dumps([object_as_dict(x) for x in entity], cls=JSONEncoder)



def get_operator_fn(op):
    return {
        '=': operator.eq,
        '>': operator.gt,
        'gt': operator.gt,
        '>=': operator.ge,
        'ge': operator.ge,
        '<': operator.lt,
        'lt': operator.lt,
        '<=': operator.le,
        'le': operator.le,
        }.get(op, operator.eq)


db = set_up_db()
app, api, ns = set_up_flask()


def return_get_dataset_parser():
    comparison_args = [
        ('id', int), ('name', str), ('train_path', str), ('test_path', str),
        ('description', str), ('n_examples', int), ('k_classes', int),
        ('d_features', int), ('majority', float), ('size_kb', int)]
    operation_args = [
        ('n_examples_op', str), ('k_classes_op', str), ('d_features_op', str),
        ('majority_op', str), ('size_kb_op', str)]

    dataset_parser = api.parser()
    for col_tuple in comparison_args:
        dataset_parser.add_argument(col_tuple[0], type=col_tuple[1])
    for col_tuple in operation_args:
        dataset_parser.add_argument(
            col_tuple[0], type=col_tuple[1],
            help='comparison operator. i.e. =, >, >=')
    return dataset_parser


def return_set_dataset_parser():
    args = [
        ('name', str), ('description', str), ('train_path', str),
        ('test_path', str), ('class_column', str), ('n_examples', int),
        ('k_classes', int), ('d_features', str), ('majority', str)]

    dataset_parser = api.parser()
    for col in args:
        dataset_parser.add_argument(col[0], type=col[1])

    return dataset_parser


get_dataset_parser = return_get_dataset_parser()
set_dataset_parser = return_set_dataset_parser()

@ns.route('/datasets')


class Dataset(Resource):
    @ns.doc('get some or all datasets')
    @api.expect(get_dataset_parser)
    def get(self):
        args = get_dataset_parser.parse_args()
        args['entity_id'] = args.get('id', None)
        args.pop('id', None)

        # deal with operations
        args['n_examples_op'] = get_operator_fn(args.get('n_examples_op', None))  # noqa
        args['k_classes_op'] = get_operator_fn(args.get('n_examples_op', None))
        args['d_features_op'] = get_operator_fn(args.get('n_examples_op', None))  # noqa
        args['majority_op'] = get_operator_fn(args.get('n_examples_op', None))
        args['size_kb_op'] = get_operator_fn(args.get('n_examples_op', None))

        res = encode_entity(db.get_datasets(**args))
        return json.loads(res)

    @ns.doc('create a dataset')
    # @api.expect(set_dataset_parser)
    def set(self):
        dataset_parser = return_set_dataset_parser()
        args = dataset_parser.parse_args()

        meta = MetaData(
            args['class_column'], args['train_path'], args['test_path'])
        args['size_kb'] = old_div(meta.size, 1000)

        dataset = db.create_dataset(**args)

        dataset = db.create_dataset(name=name,
                                    description=run_config.data_description,
                                    train_path=run_config.train_path,
                                    test_path=run_config.test_path,
                                    class_column=run_config.class_column,
                                    n_examples=meta.n_examples,
                                    k_classes=meta.k_classes,
                                    d_features=meta.d_features,
                                    majority=meta.majority,
                                    size_kb=old_div(meta.size, 1000))


if __name__ == '__main__':
    app.run(debug=True)

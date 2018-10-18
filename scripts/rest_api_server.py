from past.utils import old_div
import simplejson as json

from flask import Response
from flask_restplus import Resource

from atm.encoder import MetaData
from api.encoders import encode_entity
from api.parsers import dataset_metaparsers
from api.setup import set_up_flask


# db = set_up_db()
app, api, ns = set_up_flask()

dataset_metaparsers['get'].set_flaskplus_parser(api)
dataset_metaparsers['post'].set_flaskplus_parser(api)
dataset_metaparsers['put'].set_flaskplus_parser(api)
dataset_metaparsers['delete'].set_flaskplus_parser(api)


@ns.route('/datasets')
class Dataset(Resource):
    @ns.doc('get some or all datasets')
    @api.expect(dataset_metaparsers['get'].parser)
    def get(self):
        args = dataset_metaparsers['get'].parser.parse_args()
        args = dataset_metaparsers['get'].recode_op_args()

        try:
            res = encode_entity(
                dataset_metaparsers['get'].db.get_datasets(**args))
            return json.loads(res)
        except Exception as e:
            return json.loads(e)

    @ns.doc('create a dataset')
    @api.expect(dataset_metaparsers['post'].parser)
    def post(self):
        args = dataset_metaparsers['post'].parser.parse_args()

        meta = MetaData(
            args['class_column'], args['train_path'], args['test_path'])
        args['size_kb'] = old_div(meta.size, 1000)

        try:
            dataset = dataset_metaparsers['post'].db.create_dataset(**args)
            res = encode_entity([dataset])
            return json.loads(res)[0]
        except Exception as e:
            return json.loads(e)

    @ns.doc('update a dataset')
    @api.expect(dataset_metaparsers['post'].parser)
    def put(self):
        args = dataset_metaparsers['post'].recode_op_args()

        try:
            datasets = dataset_metaparsers['post'].db.update_datasets(**args)
            res = encode_entity(datasets)
            return json.loads(res)
        except Exception as e:
            res = json.loads(e)
            return Response(res, status=500, mimetype='application/json')

    @ns.doc('delete a dataset')
    @api.expect(dataset_metaparsers['delete'].parser)
    def delete(self):
        args = dataset_metaparsers['delete'].parser.parse_args()
        try:
            entity_id = args['entity_id']
            dataset_metaparsers['delete'].db.delete_dataset(id=entity_id)
            return Response('{}', status=201, mimetype='application/json')
        except Exception as e:
            res = json.loads(e)
            return Response(res, status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)

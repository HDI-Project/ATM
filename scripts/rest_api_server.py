from past.utils import old_div
import simplejson as json

from flask import Response, jsonify
from flask_restplus import Resource

from atm.encoder import MetaData
from api.encoders import encode_entity
from api.parsers import (dataset_metaparsers, classifier_metaparsers,
                         hyperpartition_metaparsers)
from api.setup import set_up_flask


# db = set_up_db()
app, api, ns = set_up_flask()

#############################################################################
# Hyperpartition endpoint ###################################################
#############################################################################

hyperpartition_metaparsers['get'].set_flaskplus_parser(api)


@ns.route('/hyperpartitions')
class Hyperpartitions(Resource):

    @ns.doc('get some or all hyperpartitions')
    @api.expect(hyperpartition_metaparsers['get'].parser)
    def get(self):

        args = hyperpartition_metaparsers['get'].parser.parse_args()

        try:
            res = hyperpartition_metaparsers['get'].db.get_hyperpartitions_for_api(
                **args)

            res_dict = {}
            for hype in res:
                res_dict[hype.id] = json.loads(hype.to_json())

            return jsonify(res_dict)
        except Exception as e:
            return json.loads(e)


#############################################################################
# Classifier endpoint #######################################################
#############################################################################

classifier_metaparsers['get'].set_flaskplus_parser(api)
classifier_metaparsers['post'].set_flaskplus_parser(api)
classifier_metaparsers['put'].set_flaskplus_parser(api)
classifier_metaparsers['delete'].set_flaskplus_parser(api)


@ns.route('/classifiers')
class Classifiers(Resource):

    @ns.doc('get some or all classifiers')
    @api.expect(classifier_metaparsers['get'].parser)
    def get(self):

        args = classifier_metaparsers['get'].parser.parse_args()
        args = classifier_metaparsers['get'].recode_op_args()

        try:
            # encode response, which is from a database query
            # get the object and turn it into a dict
            res = classifier_metaparsers['get'].db.get_classifiers_api(**args)
            res = encode_entity(res)
            return json.loads(res)
        except Exception as e:
            return json.loads(e)

#############################################################################
# Dataset endpoint ##########################################################
#############################################################################
# dataset metaparser is a custom class, which allows comparison operators to
# be passed


dataset_metaparsers['get'].set_flaskplus_parser(api)
dataset_metaparsers['post'].set_flaskplus_parser(api)
dataset_metaparsers['put'].set_flaskplus_parser(api)
dataset_metaparsers['delete'].set_flaskplus_parser(api)

# make a class for each endpoint


@ns.route('/datasets')
class Dataset(Resource):
    # ns.doc specifies the descriiption
    # decorate the method with its parser
    @ns.doc('get some or all datasets')
    @api.expect(dataset_metaparsers['get'].parser)
    def get(self):

        # .parser is the actual flask parser
        args = dataset_metaparsers['get'].parser.parse_args()
        args = dataset_metaparsers['get'].recode_op_args()

        try:
            # encode response, which is from a database query
            # get the object and turn it into a dict
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

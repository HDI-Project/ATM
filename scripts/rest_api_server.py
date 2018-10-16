from past.utils import old_div
import simplejson as json

from flask_restplus import Resource

from atm.encoder import MetaData
from api.encoders import encode_entity, get_operator_fn
from api.parsers import return_get_dataset_parser, return_set_dataset_parser
from api.setup import set_up_db, set_up_flask


db = set_up_db()
app, api, ns = set_up_flask()


get_dataset_parser = return_get_dataset_parser(api)
set_dataset_parser = return_set_dataset_parser(api)


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
    @api.expect(set_dataset_parser)
    def post(self):
        dataset_parser = return_set_dataset_parser()
        args = dataset_parser.parse_args()

        meta = MetaData(
            args['class_column'], args['train_path'], args['test_path'])

        args['size_kb'] = old_div(meta.size, 1000)

        dataset = db.create_dataset(**args)

        try:
            dataset = db.create_dataset(
                name=args.get('name'),
                description=args.get('data_description'),
                train_path=args.get('train_path'),
                test_path=args.get('test_path'),
                class_column=args.get('class_column'),
                n_examples=args.get('n_examples'),
                k_classes=args.get('k_classes'),
                d_features=args.get('d_features'),
                majority=args.get('majority'),
                size_kb=old_div(meta.size, 1000))

            res = encode_entity([dataset])
            return json.loads(res)[0]
        except Exception as e:
            return json.loads(e)


if __name__ == '__main__':
    app.run(debug=True)

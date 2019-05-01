from flask import abort

from atm.encoder import MetaData

DATASET_KEYS = ['name', 'description', 'train_path', 'class_column']


def dataset_post(data):
    """Preprocess the Dataset POST data."""
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
        data['size_kb'] = meta.size

    else:
        abort(400)


DATASET_PREPROCESSORS = {
    'POST': [dataset_post]
}

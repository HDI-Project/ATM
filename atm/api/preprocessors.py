import os
import traceback

from atm.api.utils import auto_abort
from atm.encoder import MetaData


@auto_abort((KeyError, FileNotFoundError))
def dataset_post(data):
    """Preprocess the Dataset POST data."""

    train_path = data['train_path']
    name = data.setdefault('name', os.path.basename(train_path))
    data.setdefault('description', name)
    meta = MetaData(
        data['class_column'],
        train_path,
        data.get('test_path')
    )

    data['n_examples'] = meta.n_examples
    data['k_classes'] = meta.k_classes
    data['d_features'] = meta.d_features
    data['majority'] = meta.majority
    data['size_kb'] = meta.size


DATASET_PREPROCESSORS = {
    'POST': [dataset_post]
}

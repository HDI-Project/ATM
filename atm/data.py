import glob
import logging
import os
import shutil

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError

LOGGER = logging.getLogger('atm')


def copy_files(extension, source, target=None):
    """Copy matching files from source to target.

    Scan the ``source`` folder and copy any file that end with
    the given ``extension`` to the ``target`` folder.

    Both ``source`` and ``target`` are expected to be either a ``str`` or a
    list or tuple of strings to be joined using ``os.path.join``.

    ``sourec`` will be interpreted as a path relative to the ``atm`` root
    code folder, and ``target`` will be interpreted as a path relative to
    the user's current working directory.

    If ``target`` is ``None``, ``source`` will be used, and if the ``target`
    directory does not exist, it will be created.

    Args:
        extension (str):
            File extension to copy.
        source (str or iterabe):
            Source directory.
        target (str or iterabe or None):
            Target directory. Defaults to ``None``.

    Returns:
        dict:
            Dictionary containing the file names without extension as keys
            and the new paths as values.
    """
    if isinstance(source, (list, tuple)):
        source = os.path.join(*source)

    if isinstance(target, (list, tuple)):
        target = os.path.join(*target)
    elif target is None:
        target = source

    source_dir = os.path.join(os.path.dirname(__file__), source)
    target_dir = os.path.join(os.getcwd(), target)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_paths = dict()

    for source_file in glob.glob(os.path.join(source_dir, '*.' + extension)):
        file_name = os.path.basename(source_file)
        target_file = os.path.join(target_dir, file_name)
        print('Generating file {}'.format(target_file))
        shutil.copy(source_file, target_file)
        file_paths[file_name[:-(len(extension) + 1)]] = target_file

    return file_paths


def get_demos():
    """Copy the demo CSV files to the ``{cwd}/demos`` folder."""
    return copy_files('csv', 'demos')


def _download_from_s3(path, local_path, aws_access_key=None, aws_secret_key=None, **kwargs):

    client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )

    bucket = path.split('/')[2]
    file_to_download = path.replace('s3://{}/'.format(bucket), '')

    try:
        LOGGER.info('Downloading {}'.format(path))
        client.download_file(bucket, file_to_download, local_path)

        return local_path

    except ClientError as e:
        LOGGER.error('An error occurred trying to download from AWS3.'
                     'The following error has been returned: {}'.format(e))


def _download_from_url(url, local_path, **kwargs):

    data = requests.get(url).text
    with open(local_path, 'wb') as outfile:
        outfile.write(data.encode())

    LOGGER.info('File saved at {}'.format(local_path))

    return local_path


DOWNLOADERS = {
    's3': _download_from_s3,
    'http': _download_from_url,
    'https': _download_from_url,
}


def _download(path, local_path, **kwargs):
    protocol = path.split(':', 1)[0]
    downloader = DOWNLOADERS.get(protocol)

    if not downloader:
        raise ValueError('Unknown protocol: {}'.format(protocol))

    return downloader(path, local_path, **kwargs)


def _get_local_path(name, path, aws_access_key=None, aws_secret_key=None):

    if os.path.isfile(path):
        return path

    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')

    if not name.endswith('csv'):
        name = name + '.csv'

    local_path = os.path.join(data_path, name)

    if os.path.isfile(local_path):
        return local_path

    if not os.path.isfile(local_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        _download(path, local_path, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key)
        return local_path


def load_data(name, path, aws_access_key=None, aws_secret_key=None):
    """Load data from the given path.

    If the path is an URL or an S3 path, download it and make a local copy
    of it to avoid having to dowload it later again.

    Args:
        name (str):
            Name of the dataset. Used to cache the data locally.
        path (str):
            Local path or S3 path or URL.
        aws_access_key (str):
            AWS access key. Optional.
        aws_secret_key (str):
            AWS secret key. Optional.

    Returns:
        pandas.DataFrame:
            The loaded data.
    """
    local_path = _get_local_path(
        name, path, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key)

    return pd.read_csv(local_path).dropna(how='any')

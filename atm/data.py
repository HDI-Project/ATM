import glob
import logging
import os
import shutil

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

LOGGER = logging.getLogger('atm')


def copy_files(pattern, source, target=None):
    if isinstance(source, (list, tuple)):
        source = os.path.join(*source)

    if target is None:
        target = source

    source_dir = os.path.join(os.path.dirname(__file__), source)
    target_dir = os.path.join(os.getcwd(), target)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    datasets = dict()

    for source_file in glob.glob(os.path.join(source_dir, pattern)):
        file_name = os.path.basename(source_file)
        target_file = os.path.join(target_dir, file_name)
        print('Generating file {}'.format(target_file))
        shutil.copy(source_file, target_file)
        datasets[file_name.replace('.csv', '')] = target_file

    return datasets


def download_demo(datasets, path=None):

    if not isinstance(datasets, list):
        datasets = [datasets]

    if path is None:
        path = os.path.join(os.getcwd(), 'demos')

    if not os.path.exists(path):
        os.makedirs(path)

    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    paths = list()

    for dataset in datasets:
        save_path = os.path.join(path, dataset)

        try:
            LOGGER.info('Downloading {}'.format(dataset))
            client.download_file('atm-data', dataset, save_path)
            paths.append(save_path)

        except ClientError as e:
            LOGGER.error('An error occurred trying to download from AWS3.'
                         'The following error has been returned: {}'.format(e))

    return paths[0] if len(paths) == 1 else paths


def get_demos(args=None):
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    available_datasets = [obj['Key'] for obj in client.list_objects(Bucket='atm-data')['Contents']]
    return available_datasets


def download_from_s3(path, local_path, aws_access_key=None, aws_secret_key=None, **kwargs):

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


def download_from_url(url, local_path, **kwargs):

    data = requests.get(url).text
    with open(local_path, 'wb') as outfile:
        outfile.write(data.encode())

    LOGGER.info('File saved at {}'.format(local_path))

    return local_path


DOWNLOADERS = {
    's3': download_from_s3,
    'http': download_from_url,
    'https': download_from_url,
}


def download(path, local_path, **kwargs):
    protocol = path.split(':', 1)[0]
    downloader = DOWNLOADERS.get(protocol)

    if not downloader:
        raise ValueError('Unknown protocol: {}'.format(protocol))

    return downloader(path, local_path, **kwargs)


def get_local_path(name, path, aws_access_key=None, aws_secret_key=None):

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

        download(path, local_path, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key)
        return local_path


def load_data(name, path, aws_access_key=None, aws_secret_key=None):
    local_path = get_local_path(
        name, path, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key)

    # load data as a Pandas dataframe
    return pd.read_csv(local_path).dropna(how='any')

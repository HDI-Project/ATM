import pandas as pd

from atm.config import CONFIG


def download_from_s3(path, local_path, **kwargs):
    aws_conf = kwargs['aws_config']


def download_from_url(url, local_path, **kwargs):
    pass


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


def get_local_path(name, path):
    if valid_path:    # TODO
        return path

    else:
        cwd = os.getcwd()
        if not name.endswith('.csv'):
            name = name + '.csv'

        return os.path.join(cwd, 'data', name)


def load_data(name, path, aws_config):
    local_path = get_local_path(name, path)

    if not os.path.isfile(local_path):
        download(path, local_path, aws_config=aws_config)

    # load data as a Pandas dataframe
    return pd.read_csv(local_path).dropna(how='any')

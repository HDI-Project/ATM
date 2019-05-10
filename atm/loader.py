import pandas as pd


def download(path):
    pass


def load_data(path, aws_conf=None):
    if not os.path.isfile(path):
        download(path)

    # load data as a Pandas dataframe
    return pd.read_csv(path).dropna(how='any')

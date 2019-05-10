import pandas as pd


def load_data(path, dropvals=None, sep=','):
    # load data as a Pandas dataframe
    data = pd.read_csv(path, skipinitialspace=True,
                       na_values=dropvals, sep=sep)

    # drop rows with any NA values
    return data.dropna(how='any')

import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from atm.utilities import ensure_directory


class MetaData(object):
    def __init__(self, label_column, *data_paths):
        """
        Compute a bunch of metadata about the dataset.

        label_column: name of dataframe column containing labels
        data_paths: paths to csvs with the same columns
        """
        data = pd.read_csv(data_paths.pop(0))
        for path in data_paths:
            data = data.append(pd.read_csv(data_path))

        # compute the portion of labels that are the most common value
        counts = data[self.label_column].value_counts()
        total_features = data.shape[1] - 1
        for c in data.columns:
            if data[c].dtype == 'object':
                total_features += np.unique(data[c]) - 1
        majority_percentage = float(max(counts)) / float(sum(counts))

        self.n_examples = data.shape[0]
        self.d_features = total_features
        self.k_classes = len(np.unique(data[label_column]))
        self.majority = majority_percentage
        self.size = np.array(data).nbytes)


class DataEncoder(object):
    def __init__(self, data_name, label_column='class',
                 testing_ratio=0.3, dropvals=None, sep=None):
        self.data_name = data_name
        self.label_column = label_column
        self.testing_ratio = testing_ratio
        self.dropvals = dropvals
        self.sep = sep or ","

        # these will be trained with fit_encoders()
        self.column_encoders = {}
        self.feature_encoder = None
        self.label_encoder = None

    def fit(self, data):
        """
        Fit one-hot encoders for categorical features and an integer encoder for
        the label. These can be used later to transform raw data into a form
        that ATM can work with.

        data: pd.DataFrame of unprocessed data
        """
        cat_cols = []
        # encode categorical columns, leave ordinal values alone
        for column in self.columns:
            if column != self.label_column and data[column].dtype == 'object':
                # save the indices of categorical columns for one-hot encoding
                cat_cols.append(data.columns.get_loc(column))

                # encode each feature as an integer in range(unique_vals)
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                self.column_encoders[column] = le

        # One-hot encode the whole feature matrix.
        # Set sparse to False so that we can test for NaNs in the output
        self.feature_encoder = OneHotEncoder(categorical_features=cat_cols,
                                             sparse=False)
        self.feature_encoder.fit(data)

        # Train an encoder for the label as well
        labels = np.array(data[[self.label_column]])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

    def transform(self, data):
        """
        Convert a DataFrame of labeled data to a feature matrix in the form
        that ATM can use.
        """
        # pull labels into a separate series and transform them to integers
        labels = np.array(data[[self.label_column]])
        y = self.label_encoder.transform(labels)

        # drop the label column and transform the remaining features
        features = data.drop([self.label_column], axis=1)

        # encode each categorical feature as an integer
        for column, encoder in self.column_encoders.items():
            data[column] = encoder.transform(data[column])

        # one-hot encode the categorical features
        X = self.feature_encoder.transform(features)

        return X, y

    def fit_transform(self, data):
        """ Process data into a form that ATM can use. """
        self.fit(data)
        return self.transform(data)

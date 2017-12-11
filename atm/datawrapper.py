import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from atm.utilities import ensure_directory


class DataWrapper(object):
    def __init__(self, dataname, output_folder, label_column, train_file,
                 test_file=None, testing_ratio=0.3, dropvals=None, sep=None):
        self.dataname = dataname
        self.output_folder = output_folder
        self.label_column = label_column
        self.train_file = train_file
        self.test_file = test_file
        self.testing_ratio = testing_ratio
        self.dropvals = dropvals
        self.sep = sep or ","

        # special objects
        self.categoricalcols = [] # names of columns that are categorical
        self.categoricalcolsidxs = []
        self.categoricalcolsvectorizers = []
        self.encoder = None # discretizes labels
        self.vectorizer = None # discretizes examples (after categoricals vectorized)

        if test_file is None:
            print "train/test data:", self.train_file
        else:
            print "training data:", self.train_file
            print "test data:", self.test_file

    @property
    def train_path_out(self):
        return os.path.join(self.output_folder, "%s_train.csv" % self.dataname)

    @property
    def test_path_out(self):
        return os.path.join(self.output_folder, "%s_test.csv" % self.dataname)

    def load_data(self, path):
        data = pd.read_csv(self.trainfile, skipinitialspace=True,
                           na_values=self.dropvals, sep=self.sep)

        # drop rows with any NA values
        data = data.dropna(how='any')

        # save the names and order of the columns for later
        self.columns = data.columns.values
        self.columns.remove(self.label_column)

        return data

    def train_encoders(self, data, labels):
        cat_cols = []
        # encode categorical columns, leave ordinal values alone
        for column in self.columns:
            if data[column].dtype == 'object':
                # save the indices of categorical columns for one-hot encoding
                cat_cols.append(data.columns.get_loc(column))

                # encode each feature as an integer in range(unique_vals)
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                self.col_encoders[column] = le

        # One-hot encode the whole feature matrix.
        # Set sparse to False so that we can test for NaNs in the output
        self.vectorizer = OneHotEncoder(categorical_features=cat_cols,
                                        sparse=False)
        self.vectorizer.fit(data)

        # Train an encoder for the label as well
        labels = np.array(data[[self.label_column]])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

    def encode_data(self, data):
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
        for column, encoder in self.col_encoders.items():
            data[column] = encoder.transform(data[column])

        # one-hot encode the categorical features
        X = self.vectorizer.transform(features)

        return np.column_stack((y, X))

    def decode_data(self, data):
        """
        Convert a DataFrame of labeled data to a feature matrix in the form
        that ATM can use.
        """

    def verify_data(self, data):
        """ Make sure everything looks good after processing """
        for column in self.columns:
            if data[column].dtype == 'object'
                # keep track of everything for verification etc.
                dummies += len(np.unique(data[column]))
                categorical += 1
            else:
                ordinal += 1

    def wrap(self):
        """ Process data into a form that ATM can use. """
        if self.test_file is not None:
            # load raw train and test data
            train_data = self.load_data(self.train_file)
            test_data = self.load_data(self.test_file)
            all_data = train_data.append(test_data, ignore_index=True)
        else:
            # load raw data and split it into train and test sets
            all_data = self.load_data(self.train_file)
            train_data, test_data = train_test_split(all_data,
                                                     test_size=self.testing_ratio)

        # train label encoder and one-hot encoders for categorical features
        self.train_encoders(all_data)

        # process data into encoded numpy arrays
        train_matrix = self.encode_data(train_data)
        test_matrix = self.encode_data(test_data)

        # save transformed data to disk
        ensure_directory(self.output_folder)
        np.savetxt(self.train_path_out, train_matrix, fmt="%s")
        np.savetxt(self.test_path_out, test_matrix, fmt="%s")

    def wrap_single_file(self):
        with open(self.traintestfile, "rb") as df:
            for line in df:
                self.headings = [x.strip() for x in line.split(self.sep)]
                break

        # now load both files and stack together
        data = pd.read_csv(self.traintestfile, skipinitialspace=True,
                           na_values=self.dropvals, sep=self.sep)
        # drop rows with any NA values
        data = data.dropna(how='any')

        # Get labels and encode into numerical values
        labels = np.array(data[[self.label_column]]) # gets data frame instead of series
        self.encoder = LabelEncoder()
        discretized_labels = self.encoder.fit_transform(labels)

        # remove labels, get majority percentage
        counts = data[self.label_column].value_counts()
        majority_percentage = float(max(counts)) / float(sum(counts))
        data = data.drop([self.label_column], axis=1) # drop column

        # get stats for database
        n, d = data.shape
        unique_classes = np.unique(labels)
        k = len(unique_classes)

        # count types of variables
        categorical = 0
        ordinal = 0
        dummies = 0

        # encode categoricals, leave ordinals alone
        for column in data.columns.values:
            if data[column].dtype == "object":
                self.categoricalcols.append(column)
                self.categoricalcolsidxs.append(data.columns.get_loc(column))

                # encode feature as an integer in range(nvalues)
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                self.categoricalcolsvectorizers.append(le)

                # keep track of everything for verification etc.
                nvalues = len(np.unique(data[column]))
                dummies += nvalues
                categorical += 1
            else:
                ordinal += 1

        # don't use sparse because then then can't test for NaNs
        self.vectorizer = OneHotEncoder(categorical_features=self.categoricalcolsidxs,
                                        sparse=False)
        data = self.vectorizer.fit_transform(data)
        newd = d - categorical + dummies

        # ensure data integrity
        # Here was the issue with this: sklearn one hot encoders will
        # convert a single array of binary variables [0,1,0,0,1] to another
        # single array, [1,0,1,1,0]. But if you give an OHE a binary variable
        # in a two-dimensional array -- e.g. [[0], [1], [0]] -- the very same
        # fit_transform will spit out TWO columns for the binary variable (one
        # of them completely redundant), e.g. [[1, 0], [0, 1], [1, 0]]. So this
        # count was off.
        assert newd == data.shape[1], "One hot encoding failed"
        assert np.sum(np.isnan(data)) == 0, \
            "Cannot have NaN values in the cleaned data!"
        assert np.sum(np.isnan(np.array(discretized_labels))) == 0, \
            "Cannot have NaN values for labels!"

        # now save training and testing as separate files
        ensure_directory(self.output_folder)

        # now split the data
        data_train, data_test, labels_train, labels_test = \
            train_test_split(data, discretized_labels,
                             test_size=self.testing_ratio)

        # training
        training_matrix = np.column_stack((labels_train, data_train))
        print "training matrix:", training_matrix.shape
        np.savetxt(self.train_path_out, training_matrix, delimiter=self.sep, fmt="%s")

        # testing
        testing_matrix = np.column_stack((labels_test, data_test))
        print "testing matrix: ", testing_matrix.shape
        np.savetxt(self.test_path_out, testing_matrix, delimiter=self.sep, fmt="%s")

        # statistics
        self.statistics = {
            "unique_classes" : list(unique_classes),
            "n_examples" : n,
            "d_features" : newd,
            "k_classes" : k,
            # this is 0 because the processed version of the data file store the
            # label class at column 0
            "label_column" : 0,
            "datasize_bytes" : np.array(data).nbytes,
            "categorical" : categorical,
            "ordinal" : ordinal,
            "dummys" : dummies,
            "majority" : majority_percentage,
            "dataname" : self.dataname,
            "training" : self.train_path_out,
            "testing" : self.test_path_out,
            "testing_ratio" : float(testing_matrix.shape[0]) /
                float(training_matrix.shape[0] + testing_matrix.shape[0]),
        }
        print 'dataset info:'
        for i in self.statistics.items():
            print '\t%s: %s' % i

    def __repr__(self):
        return "<DataWrapper: train=%r, test=%r>" % (self.train_file,
                                                     self.test_file)

    def vectorize_file(self, path):
        data = pd.read_csv(path, skipinitialspace=True,
                           na_values=self.dropvals, sep=self.sep)
        data = data.dropna(how='any') # drop rows with any NA values

        for column in data.columns.values:
            if data[column].dtype == "object":
                #self.categoricalcols.append(column)
                #self.categoricalcolsidxs.append(data.columns.get_loc(column))

                # encode feature as an integer in range(nvalues)
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                #self.categoricalcolsvectorizers.append(le)
        return self.vectorizer.transform(data)

    def vectorize_examples(self, examples):
        """
        examples: list of csv strings
            Each should be a value in the feature vector in correct order.
        """
        ready = []
        for example in examples:
            arr = [x.strip() for x in example.split(self.sep)]
            for i in range(len(arr)):
                try:
                    arr[i] = int(arr[i])
                except:
                    try:
                        arr[i] = float(arr[i])
                    except:
                        continue
            zipped = dict(zip(self.headings, arr))
            ready.append(zipped)

        return self.vectorizer.transform(ready)

    def decode_labels(self, labels):
        """
        labels: list of numerics
            Converts the list back into the original labels
        """
        return self.encoder.inverse_transform(labels)

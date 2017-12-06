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
    def __init__(self, dataname, outfolder, labelcol, testing_ratio=0.3,
                 traintestfile=None, trainfile=None, testfile=None,
                 dropvals=None, sep=None):
        self.dataname = dataname
        self.outfolder = outfolder
        self.labelcol = labelcol
        self.dropvals = dropvals
        self.sep = sep or ","
        self.testing_ratio = testing_ratio

        # special objects
        self.categoricalcols = [] # names of columns that are categorical
        self.categoricalcolsidxs = []
        self.categoricalcolsvectorizers = []
        self.encoder = None # discretizes labels
        self.vectorizer = None # discretizes examples (after categoricals vectorized)

        # can do either
        assert traintestfile or (trainfile and testfile), \
            "You must either have a single file or a train AND a test file!"

        self.traintestfile = traintestfile
        self.trainfile = trainfile
        self.testfile = testfile

        if traintestfile:
            print "train/test data:", self.traintestfile
        else:
            print "training data:", self.trainfile
            print "test data:", self.testfile


    def load(self):
        """
        Loads data from disk into trainable vector form.
        """
        pass

    def wrap(self):
        """
        Wraps data allowing for loading and vectorizing.
        """
        if self.traintestfile:
            self.wrap_single_file()
        else:
            self.wrap_train_test()
        return self.train_path_out, self.test_path_out

    def wrap_train_test(self):
        """
        """
        # get headings from first line in file
        headings = []
        with open(self.trainfile, "rb") as df:
            for line in df:
                headings = [x.strip() for x in line.split(self.sep)]
                break

        with open(self.testfile, "rb") as df:
            for line in df:
                assert headings == [x.strip() for x in line.split(self.sep)], \
                    "The heading in your testing file does not match that of your training file!"
                self.headings = headings
                break

        # now load both files and stack together
        data = pd.read_csv(self.trainfile, skipinitialspace=True,
                           na_values=self.dropvals, sep=self.sep)
        data = data.dropna(how='any')
        train_labels = np.array(data[[self.labelcol]]) # gets data frame instead of series
        self.encoder = LabelEncoder()
        training_discretized_labels = self.encoder.fit_transform(train_labels)
        num_train_samples = data.shape[0]

        # combine training and testing
        tempdata = pd.read_csv(self.testfile, skipinitialspace=True,
                               na_values=self.dropvals, sep=self.sep)
        tempdata = tempdata.dropna(how='any')
        test_labels = np.array(tempdata[[self.labelcol]]) # gets data frame instead of series
        testing_discretized_labels = self.encoder.transform(test_labels)
        num_test_samples = tempdata.shape[0]
        data = data.append(tempdata, ignore_index=True)

        # remove labels, get majority percentage
        counts = data[self.labelcol].value_counts()
        majority_percentage = float(max(counts)) / float(sum(counts))
        data = data.drop([self.labelcol], axis=1) # drop column

        # get stats for database
        n, d = data.shape
        unique_classes = np.unique(train_labels)
        k = len(unique_classes)

        # get types of variables
        categorical = 0
        ordinal = 0
        dummies = 0
        for column in data.columns.values:
            dtype = data[column].dtype
            if dtype == "object":
                self.categoricalcols.append(column)
                self.categoricalcolsidxs.append(data.columns.get_loc(column))
                nvalues = len(np.unique(np.array(data[column].values)))
                if nvalues > 2:
                    dummies += nvalues
                categorical += 1
            else:
                ordinal += 1

        for column in self.categoricalcols:
            le = LabelEncoder()
            le.fit(data[column])
            data[column] = le.transform(data[column])
            self.categoricalcolsvectorizers.append(le)

        # don't use sparse because then can't test for NaNs
        self.vectorizer = OneHotEncoder(categorical_features=self.categoricalcolsidxs,
                                        sparse=False)
        self.vectorizer.fit(data)
        data = self.vectorizer.transform(data)
        newd = d - categorical + dummies

        # ensure data integrity
        assert newd == data.shape[1], "One hot encoding failed"
        assert np.sum(np.isnan(data)) == 0, \
            "Cannot have NaN values in the cleaned data!"
        assert np.sum(np.isnan(np.array(training_discretized_labels))) == 0, \
            "Cannot have NaN values for labels!"
        assert np.sum(np.isnan(np.array(testing_discretized_labels))) == 0, \
            "Cannot have NaN values for labels!"

        # now save training and testing as separate files
        ensure_directory(self.outfolder)

        training = data[:num_train_samples,:]
        testing = data[-num_test_samples:,:]

        # training
        self.train_path_out = os.path.join(self.outfolder,
                                          "%s_train.csv" % self.dataname)
        training_matrix = np.column_stack((np.array(training_discretized_labels),
                                           np.array(training)))
        print "training matrix:", training_matrix.shape
        np.savetxt(self.train_path_out, training_matrix, delimiter=self.sep, fmt="%s")

        # testing
        self.test_path_out = os.path.join(self.outfolder,
                                          "%s_test.csv" % self.dataname)
        testing_matrix = np.column_stack((np.array(testing_discretized_labels),
                                          np.array(testing)))
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
            # TODO do something about there being both label_col and labelcol
            # and them sometimes but not always meaning the same thing
            "label_column" : 0,
            "datasize_bytes" : np.array(data).nbytes,
            "categorical" : categorical,
            "ordinal" : ordinal,
            "dummys" : dummies,
            "majority" : majority_percentage,
            "dataname" : self.dataname,
            "training" : self.train_path_out,
            "testing" : self.test_path_out,
            #"testing_ratio" : (float(testing_matrix.shape[0]) /
            #   float(training_matrix.shape[0] + testing_matrix.shape[0])),
        }
        print 'dataset info:'
        for i in self.statistics.items():
            print '\t%s: %s' % i

    def wrap_single_file(self):
        """
        """
        with open(self.traintestfile, "rb") as df:
            for line in df:
                self.headings = [x.strip() for x in line.split(self.sep)]
                break

        # now load both files and stack together
        data = pd.read_csv(self.traintestfile, skipinitialspace=True,
                           na_values=self.dropvals, sep=self.sep)
        data = data.dropna(how='any') # drop rows with any NA values

        # Get labels and encode into numerical values
        labels = np.array(data[[self.labelcol]]) # gets data frame instead of series
        self.encoder = LabelEncoder()
        discretized_labels = self.encoder.fit_transform(labels)

        # remove labels, get majority percentage
        counts = data[self.labelcol].value_counts()
        majority_percentage = float(max(counts)) / float(sum(counts))
        data = data.drop([self.labelcol], axis=1) # drop column

        # get stats for database
        n, d = data.shape
        unique_classes = np.unique(labels)
        k = len(unique_classes)

        # get types of variables
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
        ensure_directory(self.outfolder)

        # now split the data
        data_train, data_test, labels_train, labels_test = \
            train_test_split(data, discretized_labels,
                             test_size=self.testing_ratio)

        # training
        self.train_path_out = os.path.join(self.outfolder,
                                      "%s_train.csv" % self.dataname)
        training_matrix = np.column_stack((labels_train, data_train))
        print "training matrix:", training_matrix.shape
        np.savetxt(self.train_path_out, training_matrix, delimiter=self.sep, fmt="%s")

        # testing
        self.test_path_out = os.path.join(self.outfolder, "%s_test.csv" % self.dataname)
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
        return "<DataWrapper>"

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
        examples : list of csv strings
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

    def encode_labels(self, labels):
        """
        labels : list of strings
            Discretizes each value and returns as vector.
        """
        return self.encoder.transform(labels)

    def devectorize_examples(self, examples):
        """
        examples: list of numerics
            Converts the list of numerics into the original representation
        """
        return self.vectorizer.inverse_transform(examples)

    def decode_labels(self, labels):
        """
        labels: list of numerics
            Converts the list back into the original labels
        """
        return self.encoder.inverse_transform(labels)

    def feature_names(self):
        return self.vectorizer.feature_names_

    def get_statistics(self):
        return self.statistics

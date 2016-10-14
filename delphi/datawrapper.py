import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from delphi.utilities import *
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split

class DataWrapper(object):

    def __init__(self, dataname, outfolder, labelcol, testing_ratio=0.3, traintestfile=None, trainfile=None, testfile=None, 
                dropvals=None, sep=None):
        self.dataname = dataname
        self.outfolder = outfolder
        self.labelcol = labelcol
        self.dropvals = dropvals
        self.sep = sep or ","
        self.testing_ratio = testing_ratio

        # special objects
        self.encoder = None # discretizes labels
        self.vectorizer = None # discretizes examples

        # can do either
        assert traintestfile or (trainfile and testfile), \
            "You must either have a single file or a train AND a test file!"

        self.traintestfile = None
        self.trainfile = None
        self.testfile = None

        if traintestfile:
            self.traintestfile = traintestfile
        else:
            self.trainfile = trainfile
            self.testfile = testfile

        print "trainfile", self.trainfile
        print "testing", self.testfile

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
        return self.training_path, self.testing_path

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
        traindata = pd.read_csv(self.trainfile, skipinitialspace=True, na_values=self.dropvals, sep=self.sep)
        traindata = traindata.dropna(how='any')
        labels = np.array(traindata[[self.labelcol]]) # gets data frame instead of series
        self.encoder = preprocessing.LabelEncoder()
        discretized_labels = self.encoder.fit_transform(labels) 

        # combine training and testing
        testdata = pd.read_csv(self.testfile, skipinitialspace=True, na_values=self.dropvals, sep=self.sep)
        testdata = testdata.dropna(how='any')
        test_labels = np.array(testdata[[self.labelcol]]) # gets data frame instead of series
        testing_discretized_labels = self.encoder.transform(test_labels)
        alldata = traindata.append(testdata, ignore_index=True)

        # remove labels, get majority percentage
        label_col_name = alldata[alldata.columns[self.labelcol]].name
        counts = alldata[label_col_name].value_counts()
        majority_percentage = float(max(counts)) / float(sum(counts))
        alldata = alldata.drop([label_col_name], axis=1) # drop column
        testdata = testdata.drop([label_col_name], axis=1) # drop column
        traindata = traindata.drop([label_col_name], axis=1) # drop column

        # get stats for database
        n, d = alldata.shape
        unique_classes = np.unique(labels)
        k = len(unique_classes)

        # extract each row as a dictionary
        rows_as_dicts = [dict(r.iteritems()) for _, r in alldata.iterrows()]
        
        # vectorize + one hot (dummy) encoding
        self.vectorizer = DictVectorizer(sparse=False)
        self.vectorizer.fit(rows_as_dicts)
        discretized_examples = self.vectorizer.transform(rows_as_dicts)
        newd = discretized_examples.shape[1] # we've increased the dimensionaly with our encoding

        # ensure data integrity
        assert np.sum(np.isnan(np.array(discretized_examples))) == 0, \
            "Cannot have NaN values in the cleaned data!"
        assert np.sum(np.isnan(np.array(discretized_labels))) == 0, \
            "Cannot have NaN values for labels!"

        # get types of variables
        categorical = 0
        ordinal = 0
        dummies = 0
        for column in alldata.columns.values:
            dtype = alldata[column].dtype
            if dtype == "object":
                nvalues = len(np.unique(np.array(alldata[column].values)))
                if nvalues > 2:
                    dummies += nvalues
                categorical += 1
            else:
                ordinal += 1

        # now save training and testing as separate files
        EnsureDirectory(self.outfolder)

        # training
        self.training_path = os.path.join(self.outfolder, "%s_train.csv" % self.dataname)
        rows_as_dicts = [dict(r.iteritems()) for _, r in traindata.iterrows()]
        training_examples = self.vectorizer.transform(rows_as_dicts)
        training_matrix = np.column_stack((np.array(discretized_labels), np.array(training_examples)))
        print "training matrix:", training_matrix.shape
        np.savetxt(self.training_path, training_matrix, delimiter=self.sep, fmt="%s")

        # testing
        self.testing_path = os.path.join(self.outfolder, "%s_test.csv" % self.dataname)
        rows_as_dicts = [dict(r.iteritems()) for _, r in testdata.iterrows()]
        testing_examples = self.vectorizer.transform(rows_as_dicts)
        testing_matrix = np.column_stack((np.array(testing_discretized_labels), np.array(testing_examples)))
        print "testing matrix: ", testing_matrix.shape
        np.savetxt(self.testing_path, testing_matrix, delimiter=self.sep, fmt="%s")

        # statistics
        label_col = 0
        self.statistics = {
            "unique_classes" : list(unique_classes),
            "n_examples" : n, 
            "d_features" : newd, 
            "k_classes" : k, 
            "label_col" : label_col, 
            "datasize_bytes" : np.array(alldata).nbytes,
            "categorical" : categorical,
            "ordinal" : ordinal,
            "dummys" : dummies,
            "majority" : majority_percentage,
            "dataname" : self.dataname, 
            "training" : self.training_path,
            "testing" : self.testing_path,
            "testing_ratio" : float(testing_matrix.shape[0]) / float(training_matrix.shape[0] + testing_matrix.shape[0])}
        print self.statistics

    def wrap_single_file(self):
        """
        """
        with open(self.traintestfile, "rb") as df:
            for line in df:
                self.headings = [x.strip() for x in line.split(self.sep)]
                break

        # now load both files and stack together
        data = pd.read_csv(self.traintestfile, skipinitialspace=True, na_values=self.dropvals, sep=self.sep)
        data = data.dropna(how='any')
        labels = np.array(data[[self.labelcol]]) # gets data frame instead of series
        self.encoder = preprocessing.LabelEncoder()
        discretized_labels = self.encoder.fit_transform(labels) 

        # remove labels, get majority percentage
        label_col_name = data[data.columns[self.labelcol]].name
        counts = data[label_col_name].value_counts()
        majority_percentage = float(max(counts)) / float(sum(counts))
        data = data.drop([label_col_name], axis=1) # drop column

        # get stats for database
        n, d = data.shape
        unique_classes = np.unique(labels)
        k = len(unique_classes)

        # extract each row as a dictionary
        rows_as_dicts = [dict(r.iteritems()) for _, r in data.iterrows()]
        
        # vectorize + one hot (dummy) encoding
        self.vectorizer = DictVectorizer(sparse=False)
        self.vectorizer.fit(rows_as_dicts)
        discretized_examples = self.vectorizer.transform(rows_as_dicts)
        newd = discretized_examples.shape[1] # we've increased the dimensionaly with our encoding

        # ensure data integrity
        assert np.sum(np.isnan(np.array(discretized_examples))) == 0, \
            "Cannot have NaN values in the cleaned data!"
        assert np.sum(np.isnan(np.array(discretized_labels))) == 0, \
            "Cannot have NaN values for labels!"

        # get types of variables
        categorical = 0
        ordinal = 0
        dummies = 0
        for column in data.columns.values:
            dtype = data[column].dtype
            if dtype == "object":
                nvalues = len(np.unique(np.array(data[column].values)))
                if nvalues > 2:
                    dummies += nvalues
                categorical += 1
            else:
                ordinal += 1

        # now save training and testing as separate files
        EnsureDirectory(self.outfolder)

        # now split the data
        data_train, data_test, labels_train, labels_test = train_test_split(discretized_examples, discretized_labels, test_size=self.testing_ratio)
        training = pd.DataFrame(data_train)
        testing = pd.DataFrame(data_test)

        # training
        self.training_path = os.path.join(self.outfolder, "%s_train.csv" % self.dataname)
        training_matrix = np.column_stack((labels_train, data_train))
        print "training matrix:", training_matrix.shape
        np.savetxt(self.training_path, training_matrix, delimiter=self.sep, fmt="%s")

        # testing
        self.testing_path = os.path.join(self.outfolder, "%s_test.csv" % self.dataname)
        testing_matrix = np.column_stack((labels_test, data_test))
        print "testing matrix: ", testing_matrix.shape
        np.savetxt(self.testing_path, testing_matrix, delimiter=self.sep, fmt="%s")

        # statistics
        label_col = 0
        self.statistics = {
            "unique_classes" : list(unique_classes),
            "n_examples" : n, 
            "d_features" : newd, 
            "k_classes" : k, 
            "label_col" : label_col, 
            "datasize_bytes" : np.array(data).nbytes,
            "categorical" : categorical,
            "ordinal" : ordinal,
            "dummys" : dummies,
            "majority" : majority_percentage,
            "dataname" : self.dataname, 
            "training" : self.training_path,
            "testing" : self.testing_path,
            "testing_ratio" : float(testing_matrix.shape[0]) / float(training_matrix.shape[0] + testing_matrix.shape[0])}
        print self.statistics

    def __repr__(self):
        return "<DataWrapper>"

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
            ziped = dict(zip(self.headings, arr))
            ready.append(ziped)
        
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
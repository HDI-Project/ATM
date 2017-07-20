from delphi.database import *
from delphi.mapping import CreateWrapper

def get_delphi_csv_num_lines(filepath):
    with open(filepath) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def get_delphi_csv_num_cols(filepath):
    line = open(filepath).readline()
    return len(line.split(','))

# this works from the assumption the data has been preprocessed by delphi:
# no headers, numerical data only
def read_delphi_csv(filepath):
    num_rows = get_delphi_csv_num_lines(filepath)
    num_cols = get_delphi_csv_num_cols(filepath)

    data = np.zeros((num_rows, num_cols))

    with open(filepath) as f:
        for i, line in enumerate(f):
            for j, cell in enumerate(line.split(',')):
                data[i, j] = float(cell)

    return data

def LoadData(datarun):
    """
    Loads the data from HTTP (if necessary) and then from
    disk into memory.
    """
    # download data if necessary
    basepath = os.path.basename(datarun.local_trainpath)

    if not os.path.isfile(datarun.local_trainpath):
        EnsureDirectory("data/processed/")
        if not DownloadFileS3(config, datarun.local_trainpath ) == datarun.local_trainpath:
            raise Exception("Something about train dataset caching is wrong...")

    # load the data into matrix format
    trainX = read_delphi_csv(datarun.local_trainpath)
    labelcol = datarun.labelcol
    trainY = trainX[:, labelcol]
    trainX = np.delete(trainX, labelcol, axis=1)

    basepath = os.path.basename(datarun.local_testpath)
    if not os.path.isfile(datarun.local_testpath):
        EnsureDirectory("data/processed/")
        if not DownloadFileS3(config, datarun.local_testpath) == datarun.local_testpath:
            raise Exception("Something about test dataset caching is wrong...")

    # load the data into matrix format
    testX = read_delphi_csv(datarun.local_testpath)
    labelcol = datarun.labelcol
    testY = testX[:, labelcol]
    testX = np.delete(testX, labelcol, axis=1)

    return trainX, testX, trainY, testY


classifier_id = raw_input('Enter ID of classifier to be rerun: ')

if not classifier_id.isdigit():
    raise ValueError('Classifier ID must be a number!')

classifier_id = int(classifier_id)

learner = GetLearner(learner_id=classifier_id)

frozen_set = GetFrozenSet(frozen_set_id=learner.frozen_set_id, increment=False)

params = dict()

for key,val in learner.params.iteritems():
    params[key] = val

for item in frozen_set.frozens:
    params[item[0]] = item[1]


datarun = GetDatarun(datarun_id=learner.datarun_id)

wrapper = CreateWrapper(params)
trainX, testX, trainY, testY = LoadData(datarun)
wrapper.load_data_from_objects(trainX, testX, trainY, testY)

performance = wrapper.start()
for key,value in performance.iteritems():
    print '{}: {}'.format(key, value)

from delphi.database import *
from delphi.mapping import CreateWrapper, Mapping
from delphi.key import Key
import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
import argparse

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


def LoadDataFromFile(train_path, test_path, labelcol):
    # load the data into matrix format
    trainX = read_delphi_csv(train_path)
    trainY = trainX[:, labelcol]
    trainX = np.delete(trainX, labelcol, axis=1)

    # load the data into matrix format
    testX = read_delphi_csv(test_path)
    testY = testX[:, labelcol]
    testX = np.delete(testX, labelcol, axis=1)

    return trainX, testX, trainY, testY

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_params(param_str):
    param_key_value_pairs = param_str.split(',')

    params = dict()

    for pair in param_key_value_pairs:
        separated_pair = pair.split('=')

        if len(separated_pair) == 2:
            val = separated_pair[1]
            if isfloat(val):
                params[separated_pair[0]] = float(val)
            elif val.isdigit():
                params[separated_pair[0]] = int(val)
            else:
                if val == 'True':
                    params[separated_pair[0]] = True
                elif val == 'False':
                    params[separated_pair[0]] = False
                elif val == 'None':
                    params[separated_pair[0]] = None
                else:
                    params[separated_pair[0]] = separated_pair[1]

    return check_param_types(params)


def suggest_classifiers(gallery_performances, probe_performances, num_suggestions=5):
    incomplete_grid = np.vstack((gallery_performances, probe_performances))

    complete_grid = SoftImpute(max_iters=5, verbose=False).complete(incomplete_grid)

    completed_probe_performances = complete_grid[-1,:]

    suggestions = np.argsort(-completed_probe_performances) # negative so in descending order

    return suggestions[:num_suggestions]


def sample_row(row, num_non_nan_entries=5, seed=None):
    non_nan_entries = np.argwhere(np.logical_not(np.isnan(row)))

    if seed:
        np.random.seed(seed)
    selections_col_ids = np.random.choice(non_nan_entries.flatten(), size=num_non_nan_entries, replace=False)

    sampled_row = np.empty(row.shape)
    sampled_row.fill(np.nan)

    for val in selections_col_ids:
        sampled_row[val] = row[val]

    return sampled_row


def check_param_types(params):
    map = Mapping.ENUMERATOR_CODE_CLASS_MAP[params['function']]

    for param in params:
        if param in map.DEFAULT_KEYS and params[param] != None:
            key = map.DEFAULT_KEYS[param]
            if key.type == Key.TYPE_INT or key.type == Key.TYPE_INT_EXP:
                params[param] = int(params[param])
            elif key.type == Key.TYPE_BOOL:
                params[param] = bool(params[param])
            elif key.type == Key.TYPE_FLOAT or key.type == Key.TYPE_FLOAT_EXP:
                params[param] = float(params[param])

    return params


# grab the command line arguments
parser = argparse.ArgumentParser(description='Run recommender experiment')
parser.add_argument('-g', '--gridpath', help='path to grid pickle file', default=None, required=True)
parser.add_argument('-c', '--classpath', help='path to classifier dict pickle file', default=None, required=True)
parser.add_argument('-d', '--datapath', help='path to data tsv file', default=None, required=True)
parser.add_argument('-p', '--probedatasetid', help='id of probe dataset id', type=int, default=None, required=True)
parser.add_argument('-n', '--numiter', help='number of iterations', default=2, type=int, required=False)
parser.add_argument('-s', '--galsize', help='number of gallery classifiers', default=10000, type=int, required=False)
parser.add_argument('-r', '--numruns', help='number of runs', default=2, type=int, required=False)

args = parser.parse_args()

performance_file = args.gridpath
classifier_file = args.classpath
dataset_file = args.datapath

#read in grid info files
with open(performance_file, 'rb') as handle:
    grid = pickle.load(handle)

with open(classifier_file, 'rb') as handle:
    classifier_dict = pickle.load(handle)

dataset_info = pd.read_csv(dataset_file, sep='\t')

grid = grid[:, :args.galsize]

# experiment parameters
probe_dataset_id = args.probedatasetid
num_iters = args.numiter

for run_id in range(args.numruns):
    print 'starting run {}'.format(run_id)
    with open('results/dataset_{}__run_{}.csv'.format(args.probedatasetid, run_id), 'w') as f:
        # get probe dataset details
        probe_dataset_info = dataset_info[dataset_info['dataset_id'] == probe_dataset_id]
        probe_row_number = int(probe_dataset_info['row_number'])

        datarun = GetDatarun(datarun_id=probe_dataset_id, ignore_completed=False)

        # split grid into probe and gallery datasets
        probe_performances = grid[probe_row_number, :]
        gallery_performances = np.delete(grid, probe_row_number, 0)

        # sample probe performances
        sampled_probe_performance = sample_row(probe_performances, num_non_nan_entries=5)

        best_so_far = np.nanmax(sampled_probe_performance)

        f.write('Iter ID,Best Performance So Far\n')
        f.write('{},{}\n'.format(0, best_so_far))
        for iter_id in range(num_iters):
            suggestions = suggest_classifiers(gallery_performances=gallery_performances, probe_performances=sampled_probe_performance)

            for col_id in suggestions:
                if not np.isnan(probe_performances[col_id]):
                    param_str = classifier_dict.keys()[col_id]
                    params = read_params(param_str=param_str)

                    wrapper = CreateWrapper(params)
                    trainX, testX, trainY, testY = LoadData(datarun)
                    wrapper.load_data_from_objects(trainX, testX, trainY, testY)

                    performance = wrapper.start()

                    result = performance['cv_judgement_metric']
                else:
                    result = probe_performances[col_id]

                sampled_probe_performance[col_id] = result

            cur_max = np.nanmax(sampled_probe_performance)

            if cur_max > best_so_far:
                best_so_far = cur_max

            f.write('{},{}\n'.format(iter_id + 1, best_so_far))

        f.write('Best Performance From Delphi,{}\n'.format(np.nanmax(probe_performances)))
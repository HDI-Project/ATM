from btb.database import *
import numpy as np
import pickle
import warnings
import collections


def convert_classifier_to_str(classifier):
    paramstr = ''
    for key, value in classifier.params.iteritems():
        if type(value) == float:
            paramstr += '{0}={1:.5f},'.format(key, value)
        elif type(value) == int:
            paramstr += '{}={},'.format(key, value)
        elif type(value) == str:
            paramstr += '{}={},'.format(key, value)
        elif type(value) == bool:
            paramstr += '{}={},'.format(key, value)
        elif value is None:
            paramstr += '{}={},'.format(key, value)
        else:
            paramstr += '{}={},'.format(key, value)
            warnings.warn('Type unaccounted for ({})'.format(type(value)))

    return paramstr

def save_classifier_dict():
    classifier_dict = collections.OrderedDict()

    count = 0

    for datarun_id in range(1,421):
        classifiers = GetLearners(datarun_id=datarun_id)

        for classifier in classifiers:
            if classifier.is_error is not 1:
                paramstr = convert_classifier_to_str(classifier)

                if paramstr in classifier_dict.keys():
                    classifier_dict[paramstr] += 1
                else:
                    classifier_dict[paramstr] = 1

            count += 1
            if count % 1000 == 0:
                print '{} processed'.format(count)

    print '{} unique classifiers'.format(len(classifier_dict))

    with open('ordered_classifier_dict.pickle', 'wb') as handle:
        pickle.dump(classifier_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_grid():
    with open('ordered_classifier_dict.pickle', 'rb') as handle:
        classifier_dict = pickle.load(handle)

    num_unique_classifiers = len(classifier_dict)
    print '{} unique classifiers'.format(num_unique_classifiers)

    grid = np.empty((420, num_unique_classifiers))
    grid.fill(np.nan)

    count = 0
    for datarun_id in range(1, 421):
        classifiers = GetLearners(datarun_id=datarun_id)

        for classifier in classifiers:
            if classifier.is_error is not 1:
                paramstr = convert_classifier_to_str(classifier)
                col = classifier_dict.keys().index(paramstr)
                row = datarun_id - 1

                grid[row, col] = classifier.cv

            count += 1
            if count % 100 == 0:
                print '{} processed'.format(count)

    with open('grid.pickle', 'wb') as handle:
        pickle.dump(grid, handle, protocol=pickle.HIGHEST_PROTOCOL)

def export_csv_grid():
    with open('grid.pickle', 'rb') as handle:
        grid = pickle.load(handle)

    np.savetxt('grid.csv', grid, delimiter=',')

    with open('ordered_classifier_dict.pickle', 'rb') as handle:
        classifier_dict = pickle.load(handle)

    with open('classifier_definitions.tsv', 'w') as f:
        line = 'column_number\tdefinition\n'
        f.write(line)


        counter = 0
        for key in classifier_dict.keys():
            line = '{}\t{}\n'.format(counter, key)

            f.write(line)

            counter += 1

    with open('datarun_definitions.tsv', 'w') as f:
        line = 'row_number\tdataset_id\tdataset_name\n'
        f.write(line)

        counter = 0
        for datarun_id in range(1, 421):
            datarun = GetDatarun(datarun_id=datarun_id, ignore_completed=False)

            line = '{}\t{}\t{}\n'.format(counter, datarun_id, datarun.name)
            f.write(line)

            counter += 1







save_classifier_dict()
save_grid()
export_csv_grid()
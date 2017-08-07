import modelhub_api.queries as mh
import pandas as pd

classifier_ids = mh.get_classifier_ids()

classifier_structs = mh.get_classifier_details(classifier_ids)

unique_keys = dict()

for classifier_struct in classifier_structs:
    for key, value in classifier_struct.hyperparameters:

        if key not in unique_keys:
            unique_keys[key] = 1

data = pd.DataFrame(columns=unique_keys.keys())

for classifier_struct in classifier_structs:
    row = pd.DataFrame([[None, None, None, None, None, None]], columns=unique_keys.keys())

    for key, value in classifier_struct.hyperparameters:
        row[key] = value

    data = data.append(row, ignore_index=True)

data.to_csv('grid_test.csv')


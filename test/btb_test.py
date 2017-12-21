import argparse
import os
import random
from os.path import join

from atm.config import *
from atm.database import Database
from atm.enter_data import enter_datarun
from atm.utilities import download_file_s3

from utilities import *


CONF_DIR = 'config/test/btb/'
BASELINE_PATH = 'test/baselines/best_so_far/'
RUN_CONFIG = join(CONF_DIR, 'run.yaml')
SQL_CONFIG = join(CONF_DIR, 'sql.yaml')

DATA_URL = 'https://s3.amazonaws.com/mit-dai-delphi-datastore/downloaded/'

DATASETS_MAX_MIN = [
    'wholesale-customers_1.csv',
    'car_1.csv',
    'wall-robot-navigation_1.csv',
    'wall-robot-navigation_2.csv',
    'wall-robot-navigation_3.csv',
    'analcatdata_authorship_1.csv',
    'cardiotocography_1.csv',
    'wine_1.csv',
    'seismic-bumps_1.csv',
    'balance-scale_1.csv',
]
DATASETS_MAX_FIRST = [
    'wine_1.csv',
    'balance-scale_1.csv',
    'seeds_1.csv',
    'collins_1.csv',
    'cpu_1.csv',
    'vowel_1.csv',
    'car_2.csv',
    'hill-valley_2.csv',
    'rabe_97_1.csv',
    'monks-problems-2_1.csv',
]

parser = argparse.ArgumentParser(description='''
Test the performance of a new selector or tuner and compare it to that of other
methods.
''')
add_arguments_sql(parser)
add_arguments_datarun(parser)
args = parser.parse_args()

sql_conf, run_conf, _ = load_config(sql_path=SQL_CONFIG,
                                    run_path=RUN_CONFIG,
                                    args=args)
db = Database(**vars(sql_conf))
datarun_ids = {}

datasets = os.listdir(BASELINE_PATH)
datasets = datasets[:5]
print 'using datasets', ', '.join(datasets)

# generate datasets and dataruns
for ds in datasets:
    run_conf.train_path = DATA_URL + ds
    run_conf.dataset_id = None
    datarun_ids[ds] = enter_datarun(sql_conf, run_conf)

# work on the dataruns til they're done
work_parallel(db=db, datarun_ids=datarun_ids.values(), n_procs=4)

# graph the results
for ds in datasets:
    with open(join(BASELINE_PATH, ds)) as f:
        baseline = [float(l.strip()) for l in f]
    test = get_best_so_far(db, datarun_ids[ds])
    graph_series(100, ds, baseline=baseline, test=test)

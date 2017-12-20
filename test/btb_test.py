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
DATA_DIR = 'data/test/'
RUN_CONFIG = join(CONF_DIR, 'run.yaml')
SQL_CONFIG = join(CONF_DIR, 'sql.yaml')
AWS_CONFIG = join(CONF_DIR, 'aws.yaml')

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
add_arguments_aws(parser)
add_arguments_datarun(parser)
args = parser.parse_args()

sql_conf, run_conf, aws_conf = load_config(sql_path=SQL_CONFIG,
                                           run_path=RUN_CONFIG,
                                           aws_path=AWS_CONFIG,
                                           args=args)
db = Database(**vars(sql_conf))
datarun_ids = {}

datasets = os.listdir(BASELINE_PATH)
random.shuffle(datasets)
# choose a single random dataset
datasets = datasets[:5]
print 'using datasets', ', '.join(datasets)

# generate datasets and dataruns
for ds in datasets:
    # download the datset from S3
    run_conf.train_path = download_file_s3(ds, aws_conf.access_key,
                                           aws_conf.secret_key,
                                           aws_conf.s3_bucket,
                                           s3_folder=aws_conf.s3_folder,
                                           local_folder=DATA_DIR)
    run_conf.dataset_id = None
    datarun_ids[ds] = enter_datarun(sql_conf, run_conf, aws_conf)

# work on the dataruns til they're done
work_parallel(db=db, datarun_ids=datarun_ids.values(),
              aws_config=aws_conf, n_procs=4)

# graph the results
for ds in datasets:
    with open(join(BASELINE_PATH, ds)) as f:
        baseline = [float(l.strip()) for l in f]
    test = get_best_so_far(db, datarun_ids[ds])
    graph_series(100, ds, baseline=baseline, test=test)

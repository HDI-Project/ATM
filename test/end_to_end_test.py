#!/usr/bin/python2.7
import argparse
import os
import yaml
from collections import defaultdict
from os.path import join

from atm.config import *
from atm.database import Database
from atm.enter_data import enter_datarun, enter_dataset
from atm.utilities import download_file_url
from atm.worker import work

from utilities import *


CONF_DIR = 'config/test/end_to_end/'
DATA_DIR = 'data/test/'
RUN_CONFIG = join(CONF_DIR, 'run.yaml')
SQL_CONFIG = join(CONF_DIR, 'sql.yaml')
AWS_CONFIG = join(CONF_DIR, 'aws.yaml') if 'aws.yaml' in os.listdir(CONF_DIR) else None
S3_BUCKET = 'https://s3.amazonaws.com/mit-dai-delphi-datastore/downloaded/'

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
DATASETS_SIMPLE = [
    'vowel_1.csv',
    #'iris.data.csv',
    #'multilabeltest.csv',
    #'bigmultilabeltest.csv',
]

DATASETS = DATASETS_SIMPLE


parser = argparse.ArgumentParser(description='''
Run a single end-to-end test with 10 sample datasets.
The script will create a datarun for each dataset, then run a worker until the
jobs are finished.
''')
parser.add_argument('--processes', help='number of processes to run concurrently',
                    type=int, default=4)

args = parser.parse_args()
sql_config, run_config, aws_config = load_config(sql_path=SQL_CONFIG,
                                                 run_path=RUN_CONFIG,
                                                 aws_path=AWS_CONFIG)
db = Database(**vars(sql_config))

print 'creating dataruns...'
datarun_ids = []
for ds in DATASETS:
    # remove existing and redownload data to ensure clasifiers are generated
    if ds in os.listdir(DATA_DIR):
        os.remove(DATA_DIR+ds)
    run_config.train_path = download_file_url(S3_BUCKET+ds, DATA_DIR)
    dataset = enter_dataset(db, run_config, aws_config)
    datarun_ids.append(enter_datarun(sql_config, run_config, aws_config))

work_parallel(db=db, datarun_ids=datarun_ids, aws_config=aws_config,
              n_procs=args.processes)

print 'workers finished.'

for rid in datarun_ids:
    print_summary(db, rid)

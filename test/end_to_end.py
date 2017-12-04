#!/usr/bin/python2.7
import argparse
import os
import yaml

from atm.database import Database
from atm.enter_data import load_config, enter_data
from atm.utilities import download_file_s3
from atm.worker import work


RUN_CONFIG = 'config/test/run_config.yaml'
SQL_CONFIG = 'config/test/sql_config.yaml'
AWS_CONFIG = 'config/test/aws_config.yaml'
DATA_PATH = 'data/test/'
DATASETS_MAX_MIN = ['wholesale-customers_1.csv',
                    'car_1.csv',
                    'wall-robot-navigation_1.csv',
                    'wall-robot-navigation_2.csv',
                    'wall-robot-navigation_3.csv',
                    'analcatdata_authorship_1.csv',
                    'cardiotocography_1.csv',
                    'wine_1.csv',
                    'seismic-bumps_1.csv',
                    'balance-scale_1.csv']
DATASETS_MAX_FIRST = ['wine_1.csv',
                      'balance-scale_1.csv',
                      'seeds_1.csv',
                      'collins_1.csv',
                      'cpu_1.csv',
                      'vowel_1.csv',
                      'car_2.csv',
                      'hill-valley_2.csv',
                      'rabe_97_1.csv',
                      'monks-problems-2_1.csv']
DATASETS_SIMPLE = []

DATASETS = DATASETS_MAX_FIRST


parser = argparse.ArgumentParser(description='''
Run a single end-to-end test with 10 sample datasets.
The script will create a datarun for each dataset, then run a worker until the
jobs are finished.
''')
parser.add_argument('--aws-config', help='path to AWS configuration',
                    default=AWS_CONFIG)

args = parser.parse_args()

print 'creating dataruns...'
sql_config, aws_config, run_config = load_config(sql_path=SQL_CONFIG,
                                                 aws_path=args.aws_config,
                                                 run_path=RUN_CONFIG)

datarun_ids = []
for ds in DATASETS:
    # download the datset from S3
    download_file_s3(ds, aws_config.access_key, aws_config.secret_key,
                     aws_config.s3_bucket, s3_folder=aws_config.s3_folder,
                     local_folder=DATA_PATH)
    run_config.train_path = os.path.join(DATA_PATH, ds)
    run_config.dataset_id = None
    datarun_ids.append(enter_data(sql_config, aws_config, run_config))

print 'starting workers...'

# TODO multicore
work(Database(**vars(sql_config)), datarun_ids=datarun_ids, save_files=False,
     choose_randomly=True, cloud_mode=False, aws_config=aws_config)

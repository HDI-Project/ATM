#!/usr/bin/python2.7
import argparse
import os
import yaml

from atm.database import Database
from atm.enter_data import load_config, enter_data
from atm.utilities import download_file_s3
from atm.worker import Worker


RUN_CONFIG = 'config/test/run_config.yaml'
SQL_CONFIG = 'config/test/sql_config.yaml'
AWS_CONFIG = 'config/test/aws_config.yaml'
DATA_PATH = 'data/test/'
DATASETS = ['wine_1.csv',
            'balance-scale_1.csv',
            'seeds_1.csv',
            'collins_1.csv',
            'cpu_1.csv',
            'vowel_1.csv',
            'car_2.csv',
            'hill-valley_2.csv',
            'rabe_97_1.csv',
            'monks-problems-2_1.csv']


parser = argparse.ArgumentParser(description='''
Run a single end-to-end test with 10 sample datasets.
The script will create a datarun for each dataset, then run a worker until the
jobs are finished.
''')
parser.add_argument('--aws-config', help='path to AWS configuration',
                    default=AWS_CONFIG)

args = parser.parse_args()

print 'creating dataruns...'
config = load_config(sql_config=SQL_CONFIG, aws_config=args.aws_config,
                     run_config=RUN_CONFIG)

for ds in DATASETS:
    # download the datset from S3
    download_file_s3(ds, config.aws_access_key, config.aws_secret_key,
                     config.aws_s3_bucket, s3_folder=config.aws_s3_folder,
                     outpath=DATA_PATH)
    config.train_path = os.path.join(DATA_PATH, ds)
    config.dataset_id = None
    datarun_id = enter_data(config)

print 'starting worker...'

with open(SQL_CONFIG) as f:
    sql_config = yaml.load(f)
db = Database(**sql_config)

with open(args.aws_config) as f:
    aws_config = yaml.load(f)

worker = Worker(db, datarun_id=None, save_files=False, choose_randomly=True,
                cloud_mode=False, aws_config=aws_config)
worker.work()

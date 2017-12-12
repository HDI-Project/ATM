#!/usr/bin/python2.7
import argparse
import os
import yaml
from collections import defaultdict
from os.path import join

from atm.config import *
from atm.database import Database
from atm.enter_data import enter_datarun, enter_dataset
from atm.utilities import download_file_s3
from atm.worker import work

from utilities import *


CONF_DIR = 'config/test/method/'
DATA_DIR = 'data/test/'
RUN_CONFIG = join(CONF_DIR, 'run.yaml')
SQL_CONFIG = join(CONF_DIR, 'sql.yaml')
AWS_CONFIG = join(CONF_DIR, 'aws.yaml')
DATASETS = [
    'iris.data.csv',
    #'multilabeltest.csv',
    #'bigmultilabeltest.csv',
]


parser = argparse.ArgumentParser(description='''
Run a single end-to-end test with 10 sample datasets.
The script will create a datarun for each dataset, then run a worker until the
jobs are finished.
''')
parser.add_argument('--processes', help='number of processes to run concurrently',
                    type=int, default=1)
parser.add_argument('--method', help='code for method to test')
parser.add_argument('--method-json', help='path to config for method to test')

args = parser.parse_args()
sql_config, run_config, aws_config = load_config(sql_path=SQL_CONFIG,
                                                 run_path=RUN_CONFIG,
                                                 aws_path=AWS_CONFIG)
db = Database(**vars(sql_config))

print 'creating dataruns...'
datarun_ids = []
for ds in DATASETS:
    run_config.train_path = join(DATA_DIR, ds)
    run_config.methods = [args.method]
    dataset = enter_dataset(db, run_config, aws_config)
    datarun_ids.append(enter_datarun(sql_config, run_config, aws_config))

work_parallel(db=db, datarun_ids=datarun_ids, aws_config=aws_config,
              n_procs=args.processes)

print 'workers finished.'

for rid in datarun_ids:
    print_summary(db, rid)

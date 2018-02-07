#!/usr/bin/python2.7
from __future__ import print_function
import argparse
import os
import yaml
from collections import defaultdict
from os.path import join

from atm.config import *
from atm.database import Database
from atm.enter_data import enter_data
from atm.utilities import download_file_s3
from atm.worker import work

from utilities import *


CONF_DIR = os.path.join(PROJECT_ROOT, 'config/test/')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/test/')
RUN_CONFIG = join(CONF_DIR, 'run-default.yaml')
SQL_CONFIG = join(CONF_DIR, 'sql-sqlite.yaml')
DATASETS = [
    'iris.data.csv',
    'pollution_1.csv',
]


parser = argparse.ArgumentParser(description='''
Run a single end-to-end test with 10 sample datasets.
The script will create a datarun for each dataset, then run a worker until the
jobs are finished.
''')
parser.add_argument('--processes', help='number of processes to run concurrently',
                    type=int, default=1)
parser.add_argument('--method', help='code for method to test')
parser.add_argument('--method-path', help='path to JSON config for method to test')

args = parser.parse_args()
sql_config, run_config, aws_config, _ = load_config(sql_path=SQL_CONFIG,
                                                    run_path=RUN_CONFIG)
db = Database(**vars(sql_config))

print('creating dataruns...')
datarun_ids = []
for ds in DATASETS:
    run_config.train_path = join(DATA_DIR, ds)
    if args.method:
        run_config.methods = [args.method]
    else:
        run_config.methods = METHODS
    datarun_ids.extend(enter_data(sql_config, run_config, aws_config,
                                  run_per_partition=True))

print('computing on dataruns', datarun_ids)
work_parallel(db=db, datarun_ids=datarun_ids, aws_config=aws_config,
              n_procs=args.processes)

print('workers finished.')

for rid in datarun_ids:
    print_hp_summary(db, rid)

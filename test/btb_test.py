from __future__ import print_function
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
RUN_CONFIG = join(CONF_DIR, 'run.yaml')
SQL_CONFIG = join(CONF_DIR, 'sql.yaml')

DATA_URL = 'https://s3.amazonaws.com/mit-dai-delphi-datastore/downloaded/'

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
parser.add_argument('--processes', help='number of processes to run concurrently',
                    type=int, default=1)
parser.add_argument('--tuner', type=option_or_path(TUNERS),
                    help='type of, or path to, BTB tuner')
parser.add_argument('--selector', type=option_or_path(SELECTORS),
                    help='type of, or path to, BTB selector')
parser.add_argument('--graph', action='store_true', default=False,
                    help='if this flag is inculded, graph the best-so-far '
                    'results of each datarun against the baseline.')
parser.add_argument('--dataruns', nargs='+', type=int,
                    help='(optional) IDs of previously-created dataruns to graph. '
                    'If this option is included, no new dataruns will be created.')
args = parser.parse_args()

sql_conf, run_conf, _ = load_config(sql_path=SQL_CONFIG,
                                    run_path=RUN_CONFIG,
                                    args=args)
db = Database(**vars(sql_conf))
datarun_ids = args.dataruns or []
baselines = os.listdir(BASELINE_PATH)
datasets = [d for d in DATASETS_MAX_FIRST if d in baselines]

# if necessary, generate datasets and dataruns
if not datarun_ids:
    for ds in datasets:
        run_conf.train_path = DATA_URL + ds
        run_conf.dataset_id = None
        print('Creating datarun for', run_conf.train_path)
        datarun_ids.append(enter_datarun(sql_conf, run_conf))

# work on the dataruns til they're done
print('Working on %d dataruns' % len(datarun_ids))
work_parallel(db=db, datarun_ids=datarun_ids, n_procs=args.processes)
print('Finished!')

# graph the results
for rid in datarun_ids:
    report_auc_vs_baseline(db, rid, graph=args.graph)

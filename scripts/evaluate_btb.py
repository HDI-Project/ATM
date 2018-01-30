from __future__ import print_function
import argparse
import os
import random
from os.path import join

from atm import PROJECT_ROOT
from atm.config import *
from atm.database import Database
from atm.enter_data import enter_data

from utilities import *


CONF_DIR = os.path.join(PROJECT_ROOT, 'config/test/')
RUN_CONFIG = join(CONF_DIR, 'run-default.yaml')
SQL_CONFIG = join(CONF_DIR, 'sql-sqlite.yaml')

DATASETS_MAX_FIRST = [
    'collins_1.csv',
    'cpu_1.csv',
    'vowel_1.csv',
    'car_2.csv',
    'hill-valley_2.csv',
    'rabe_97_1.csv',
    'monks-problems-2_1.csv',
    # these datasets do not have baseline numbers
    #'wine_1.csv',
    #'balance-scale_1.csv',
    #'seeds_1.csv',
]


def btb_test(dataruns=None, datasets=None, processes=1, graph=False, **kwargs):
    """
    Run a test datarun using the chosen tuner and selector, and compare it to
    the baseline performance.

    Tuner and selector will be specified in **kwargs, along with the rest of the
    standard datarun arguments.
    """
    sql_conf, run_conf, _ = load_config(sql_path=SQL_CONFIG,
                                        run_path=RUN_CONFIG,
                                        **kwargs)

    db = Database(**vars(sql_conf))
    datarun_ids = dataruns or []
    datarun_ids_per_dataset = [[each] for each in dataruns] if dataruns else []
    datasets = datasets or DATASETS_MAX_FIRST

    # if necessary, generate datasets and dataruns
    if not datarun_ids:
        for ds in datasets:
            run_conf.train_path = DATA_URL + ds
            run_conf.dataset_id = None
            print('Creating 10 dataruns for', run_conf.train_path)
            run_ids = [enter_data(sql_conf, run_conf) for i in range(10)]
            datarun_ids_per_dataset.append(run_ids)
            datarun_ids.extend(run_ids)

    # work on the dataruns til they're done
    print('Working on %d dataruns' % len(datarun_ids))
    work_parallel(db=db, datarun_ids=datarun_ids, n_procs=processes)
    print('Finished!')

    results = {}

    # compute and maybe graph the results for each dataset
    for rids in datarun_ids_per_dataset:
        res = report_auc_vs_baseline(db, rids, graph=graph)
        results[tuple(rids)] = {'test': res[0], 'baseline': res[1]}

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Test the performance of an AutoML method and compare it to the baseline
    performance curve.
    ''')
    parser.add_argument('--processes', help='number of processes to run concurrently',
                        type=int, default=1)
    parser.add_argument('--graph', action='store_true', default=False,
                        help='if this flag is inculded, graph the best-so-far '
                        'results of each datarun against the baseline.')
    parser.add_argument('--dataruns', nargs='+', type=int,
                        help='(optional) IDs of previously-created dataruns to '
                        'graph. If this option is included, no new dataruns '
                        'will be created, but any of the specified dataruns '
                        'will be finished if they are not already.')
    parser.add_argument('--datasets', nargs='+',
                        help='(optional) file names of training data to use. '
                        'Each should be a csv file present in the downloaded/ '
                        'folder of the HDI project S3 bucket '
                        '(https://s3.amazonaws.com/mit-dai-delphi-datastore/downloaded/).'
                        'The default is to use the files in DATASETS_MAX_FIRST.')
    add_arguments_datarun(parser)
    args = parser.parse_args()

    btb_test(**vars(args))

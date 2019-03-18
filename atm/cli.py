# -*- coding: utf-8 -*-

import argparse
import os

from atm import PROJECT_ROOT
from atm.config import (
    add_arguments_aws_s3, add_arguments_datarun, add_arguments_logging, add_arguments_sql,
    initialize_logging, load_config)
from atm.database import Database
from atm.enter_data import enter_data
from atm.worker import work

CONF_DIR = os.path.join(PROJECT_ROOT, 'config/test/')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/test/')
RUN_CONFIG = os.path.join(CONF_DIR, 'run-all.yaml')
SQL_CONFIG = os.path.join(CONF_DIR, 'sql-sqlite.yaml')

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
    'pollution_1.csv',  # binary test data
    'iris.data.csv',    # ternary test data
]


def _end_to_end_test(args):
    """End to end test"""

    db = Database(**vars(sql_config))

    if args.verbose:
        print('creating dataruns...')

    datarun_ids = []

    for ds in DATASETS_SIMPLE:
        run_config.train_path = os.path.join(DATA_DIR, ds)
        datarun_ids.append(enter_data(sql_config=sql_config, run_config=run_config))

    work_parallel(
        db=db,
        datarun_ids=datarun_ids,
        n_procs=args.processes,
        total_time=args.total_time
    )

    for rid in datarun_ids:
        print_summary(db, rid)


# default values, user values
def _work(args):

    # default logging config is different if initialized from the command line
    if args.log_config is None:
        args.log_config = os.path.join(PROJECT_ROOT,
                                       'config/templates/log-script.yaml')

    sql_config, _, aws_config, log_config = load_config(**vars(args))
    initialize_logging(log_config)

    # let's go
    work(db=Database(**vars(sql_config)),
         datarun_ids=args.dataruns,
         choose_randomly=args.choose_randomly,
         save_files=args.save_files,
         cloud_mode=args.cloud_mode,
         aws_config=aws_config,
         log_config=log_config,
         total_time=args.time,
         wait=False)


def _enter_data(args):

    # default logging config is different if initialized from the command line
    if args.log_config is None:
        args.log_config = os.path.join(PROJECT_ROOT,
                                       'config/templates/log-script.yaml')

    # create config objects from the config files and/or command line args
    sql_conf, run_conf, aws_conf, log_conf = load_config(sql_path=args.sql_config,
                                                         run_path=args.run_config,
                                                         aws_path=args.aws_config,
                                                         log_path=args.log_config,
                                                         **vars(args))
    initialize_logging(log_conf)

    # create and save the dataset and datarun
    enter_data(sql_conf, run_conf, aws_conf, args.run_per_partition)


# load other functions from config.py
def _add_common_arguments(parser):
    add_arguments_sql(parser)
    add_arguments_aws_s3(parser)
    add_arguments_logging(parser)


def _get_parser():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Be verbose')

    parser = argparse.ArgumentParser(description='ATM Command Line Interface')

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    # Enter Data Parser
    enter_data = subparsers.add_parser('enter_data', parents=[parent])
    enter_data.set_defaults(action=_enter_data)
    _add_common_arguments(enter_data)
    add_arguments_datarun(enter_data)
    enter_data.add_argument('--run-per-partition', default=False, action='store_true',
                            help='if set, generate a new datarun for each hyperpartition')

    # Worker
    worker = subparsers.add_parser('worker', parents=[parent])
    worker.set_defaults(action=_work)
    worker.add_argument('--cloud-mode', action='store_true', default=False,
                        help='Whether to run this worker in cloud mode')

    worker.add_argument('--dataruns', help='Only train on dataruns with these ids', nargs='+')
    worker.add_argument('--time', help='Number of seconds to run worker', type=int)
    worker.add_argument('--choose-randomly', action='store_true',
                        help='Choose dataruns to work on randomly (default = sequential order)')

    worker.add_argument('--no-save', dest='save_files', default=True,
                        action='store_const', const=False,
                        help="don't save models and metrics at all")

    # End to end test
    end_to_end = subparser.add_parser('end_to_end', parents=[parent])
    end_to_end.set_defaults(action=_end_to_end_test)
    end_to_end.add_argument('--processes', help='number of processes to run concurrently',
                            type=int, default=4)

    end_to_end.add_argument('--total-time', help='Total time for each worker to work in seconds.',
                            type=int, default=None)




def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)

# -*- coding: utf-8 -*-

import argparse
import inspect
import os

from atm import PROJECT_ROOT
from atm.config import (add_arguments_aws_s3, add_arguments_sql,
                        add_arguments_datarun, add_arguments_logging,
                        load_config, initialize_logging)

from atm.database import Database
from atm.enter_data import enter_data
from atm.worker import work


# database cli

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
    enter_data = subparsers.add_parser('enter_data', parents=[parents])
    enter_data.set_defaults(action=_enter_data)
    _add_common_arguments(enter_data)
    add_arguments_datarun(enter_data)
    enter_data.add_argument('--run-per-partition', default=False, action='store_true',
                            help='if set, generate a new datarun for each hyperpartition')

    # Worker
    worker = subparsers.add_parser('worker', parents=[parents])
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


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)

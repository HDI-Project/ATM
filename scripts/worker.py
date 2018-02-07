#!/usr/bin/python2.7
from __future__ import print_function

import argparse
import datetime
import warnings

from atm.config import (add_arguments_aws_s3, add_arguments_logging,
                        add_arguments_sql, load_config, initialize_logging)
from atm.database import Database
from atm.worker import Worker, work

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add more classifiers to database')
    add_arguments_sql(parser)
    add_arguments_aws_s3(parser)
    add_arguments_logging(parser)

    # add worker-specific arguments
    parser.add_argument('--cloud-mode', action='store_true', default=False,
                        help='Whether to run this worker in cloud mode')
    parser.add_argument('--dataruns', help='Only train on dataruns with these ids',
                        nargs='+')
    parser.add_argument('--time', help='Number of seconds to run worker', type=int)
    parser.add_argument('--choose-randomly', action='store_true',
                        help='Choose dataruns to work on randomly (default = sequential order)')
    parser.add_argument('--no-save', dest='save_files', default=True,
                        action='store_const', const=False,
                        help="don't save models and metrics at all")

    # parse arguments and load configuration
    args = parser.parse_args()
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

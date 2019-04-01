# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil

from atm.config import (
    add_arguments_aws_s3, add_arguments_datarun, add_arguments_logging, add_arguments_sql)
from atm.models import ATM


def _end_to_end_test(args):
    """End to end test"""


def _work(args):
    atm = ATM(**vars(args))
    atm.work(
        datarun_ids=args.dataruns,
        choose_randomly=args.choose_randomly,
        save_files=args.save_files,
        cloud_mode=args.cloud_mode,
        total_time=args.time,
        wait=False
    )


def _enter_data(args):
    atm = ATM(**vars(args))
    atm.enter_data()


def _make_config(args):
    config_templates = os.path.join('config', 'templates')
    config_dir = os.path.join(os.path.dirname(__file__), config_templates)
    target_dir = os.path.join(os.getcwd(), config_templates)
    os.makedirs(target_dir, exist_ok=True)
    for template in glob.glob(os.path.join(config_dir, '*.yaml')):
        target_file = os.path.join(target_dir, os.path.basename(template))
        print('Generating file {}'.format(target_file))
        shutil.copy(template, target_file)


# load other functions from config.py
def _add_common_arguments(parser):
    add_arguments_sql(parser)
    add_arguments_aws_s3(parser)
    add_arguments_logging(parser)


def _get_parser():
    parent = argparse.ArgumentParser(add_help=False)

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
    _add_common_arguments(worker)
    worker.add_argument('--cloud-mode', action='store_true', default=False,
                        help='Whether to run this worker in cloud mode')

    worker.add_argument('--dataruns', help='Only train on dataruns with these ids', nargs='+')
    worker.add_argument('--time', help='Number of seconds to run worker', type=int)
    worker.add_argument('--choose-randomly', action='store_true',
                        help='Choose dataruns to work on randomly (default = sequential order)')

    worker.add_argument('--no-save', dest='save_files', default=True,
                        action='store_const', const=False,
                        help="don't save models and metrics at all")

    # Make Config
    make_config = subparsers.add_parser('make_config', parents=[parent])
    make_config.set_defaults(action=_make_config)

    # End to end test
    end_to_end = subparsers.add_parser('end_to_end', parents=[parent])
    end_to_end.set_defaults(action=_end_to_end_test)
    end_to_end.add_argument('--processes', help='number of processes to run concurrently',
                            type=int, default=4)

    end_to_end.add_argument('--total-time', help='Total time for each worker to work in seconds.',
                            type=int, default=None)

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)

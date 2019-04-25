# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import shutil
import socket
import time
from multiprocessing import Pool, Process, Queue

from atm.api import create_app
from atm.config import (
    add_arguments_aws_s3, add_arguments_datarun, add_arguments_logging, add_arguments_sql,
    load_config)
from atm.database import Database
from atm.models import ATM
from atm.worker import ClassifierError, Worker

LOGGER = logging.getLogger(__name__)


def _get_db(args):
    db_args = {
        k[4:]: v
        for k, v in vars(args).items()
        if k.startswith('sql_') and v is not None
    }
    return Database(**db_args)



def _serve(args):
    db = _get_db(args)
    app = create_app(db)
    app.run(host=args.host, port=args.port)


def _get_next_datarun(db):
    dataruns = db.get_dataruns(ignore_complete=True)
    if dataruns:
        max_priority = max([datarun.priority for datarun in dataruns])
        priority_runs = [r for r in dataruns if r.priority == max_priority]
        return priority_runs[0]


def _process_datarun(args, queue):
    run_conf, aws_conf, log_conf = load_config(**vars(args))
    db = _get_db(args)

    while True:
        datarun_id = queue.get(True)

        dataruns = db.get_dataruns(include_ids=[datarun_id])
        if dataruns:
            datarun = dataruns[0]

            worker = Worker(db, datarun, save_files=args.save_files,
                            cloud_mode=args.cloud_mode, aws_config=aws_conf,
                            log_config=log_conf, public_ip=socket.gethostname())

            try:
                worker.run_classifier()

            except ClassifierError:
                # the exception has already been handled; just wait a sec so we
                # don't go out of control reporting errors
                LOGGER.warning('Something went wrong. Sleeping %d seconds.', 1)
                time.sleep(1)


def _worker_loop(args):
    db = _get_db(args)

    queue = Queue(1)
    LOGGER.info('Starting %s worker processes', args.workers)
    with Pool(args.workers, _process_datarun, (args, queue, )):
        while True:
            datarun = _get_next_datarun(db)

            if not datarun:
                time.sleep(1)
                continue

            LOGGER.warning('Processing datarun %d', datarun.id)
            db.mark_datarun_running(datarun.id)

            queue.put(datarun.id)


def _start(args):
    if args.server:
        LOGGER.info('Starting the REST API server')
        process = Process(target=_serve, args=(args, ))
        process.daemon = True
        process.start()

    _worker_loop(args)


def _enter_data(args):
    db = _get_db(args)
    run_conf, aws_conf, log_conf = load_config(**vars(args))
    atm = ATM(db, run_conf, aws_conf, log_conf)

    atm.enter_data()


def _make_config(args):
    config_templates = os.path.join('config', 'templates')
    config_dir = os.path.join(os.path.dirname(__file__), config_templates)
    target_dir = os.path.join(os.getcwd(), config_templates)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

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
    parent.add_argument('-v', '--verbose', action='count', default=0)
    parent.add_argument('-l', '--logfile')

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

    # Start
    start = subparsers.add_parser('start', parents=[parent])
    start.set_defaults(action=_start)
    _add_common_arguments(start)
    start.add_argument('--cloud-mode', action='store_true', default=False,
                       help='Whether to run this worker in cloud mode')
    start.add_argument('--no-save', dest='save_files', default=True,
                       action='store_const', const=False,
                       help="don't save models and metrics at all")
    start.add_argument('-w', '--workers', default=1, type=int, help='Number of workers')

    start.add_argument('--server', action='store_true',
                       help='Also start the REST server')
    start.add_argument('--host', help='IP to listen at')
    start.add_argument('--port', help='Port to listen at', type=int)

    # Make Config
    make_config = subparsers.add_parser('make_config', parents=[parent])
    make_config.set_defaults(action=_make_config)

    return parser


def _logging_setup(verbosity=1, logfile=None):
    logger = logging.getLogger()
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    _logging_setup(args.verbose, args.logfile)

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)

# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import multiprocessing
import os
import shutil
import socket
import time

import psutil
from daemon import DaemonContext
from lockfile.pidlockfile import PIDLockFile

from atm.api import create_app
from atm.config import (
    add_arguments_aws_s3, add_arguments_datarun, add_arguments_logging, add_arguments_sql,
    load_config)
from atm.database import Database
from atm.models import ATM
from atm.worker import ClassifierError, Worker

LOGGER = logging.getLogger(__name__)


def _get_db(args):
    """Returns an instance of Database with the given args."""
    db_args = {
        k[4:]: v
        for k, v in vars(args).items()
        if k.startswith('sql_') and v is not None
    }
    return Database(**db_args)


def _work(args):
    """Creates a single worker on the current terminal / window."""
    db = _get_db(args)
    run_conf, aws_conf, log_conf = load_config(**vars(args))

    atm = ATM(db, run_conf, aws_conf, log_conf)

    atm.work(
        datarun_ids=args.dataruns,
        choose_randomly=False,
        save_files=args.save_files,
        cloud_mode=args.cloud_mode,
        total_time=args.time,
        wait=False
    )


def _serve(args):
    """Launch the ATM API with the given host / port."""
    db = _get_db(args)
    app = create_app(db, args.debug)
    app.run(host=args.host, port=args.port)


def _get_next_datarun(db):
    """Get the following datarun with the max priority."""
    dataruns = db.get_dataruns(ignore_complete=True)
    if dataruns:
        max_priority = max([datarun.priority for datarun in dataruns])
        priority_runs = [r for r in dataruns if r.priority == max_priority]
        return priority_runs[0]


def _process_datarun(args, queue):
    """Process the datarun with the worker."""
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
    """We create a multiprocessing Queue and then a pool with the number of workers specified
    by the args which stay on a loop listening for new entries inside the database.
    """
    db = _get_db(args)

    queue = multiprocessing.Queue(1)
    LOGGER.info('Starting %s worker processes', args.workers)
    with multiprocessing.Pool(args.workers, _process_datarun, (args, queue, )):
        while True:
            datarun = _get_next_datarun(db)

            if not datarun:
                time.sleep(1)
                continue

            LOGGER.warning('Processing datarun %d', datarun.id)
            db.mark_datarun_running(datarun.id)

            queue.put(datarun.id)


def _get_pid_path(pid):
    """Returns abspath of the pid file which is stored on the cwd."""
    pid_path = pid

    if not os.path.isabs(pid_path):
        pid_path = os.path.join(os.getcwd(), pid_path)

    return pid_path


def _get_atm_process(pid_path):
    """Return `psutil.Process` of the `pid` file. If the pidfile is stale it will release it."""
    pid_file = PIDLockFile(pid_path)

    if pid_file.is_locked():
        pid = pid_file.read_pid()

        try:
            process = psutil.Process(pid)
            if process.name() == 'atm':
                return process
            else:
                pid_file.break_lock()

        except psutil.NoSuchProcess:
            pid_file.break_lock()


def _status(args):
    """Check if the current ATM process is runing."""

    pid_path = _get_pid_path(args.pid)
    process = _get_atm_process(pid_path)

    if process:
        workers = 0
        addr = None
        for child in process.children():
            connections = child.connections()
            if connections:
                connection = connections[0]
                addr = connection.laddr

            else:
                workers += 1

        s = 's' if workers > 1 else ''
        print('ATM is running with {} worker{}'.format(workers, s))

        if addr:
            print('ATM REST server is listening on http://{}:{}'.format(addr.ip, addr.port))

    else:
        print('ATM is not runing.')


def _start_background(args):
    """Launches the server/worker in daemon process."""
    if not args.no_server:
        LOGGER.info('Starting the REST API server')

        process = multiprocessing.Process(target=_serve, args=(args, ))
        process.daemon = True

        process.start()

    if args.workers:
        _worker_loop(args)


def _start(args):
    """Create a new process of ATM pointing the process to a certain `pid` file."""
    pid_path = _get_pid_path(args.pid)
    process = _get_atm_process(pid_path)

    if process:
        print('ATM is already running!')

    else:
        pid_file = PIDLockFile(pid_path)

        context = DaemonContext()
        context.pidfile = pid_file
        context.working_directory = os.getcwd()

        with context:
            # Set up default logs
            if not args.logfile:
                _logging_setup(args.verbose, 'atm.log')

            print('Starting ATM')
            _start_background(args)


def _restart(args):
    if _stop(args):
        time.sleep(1)
        _start(args)


def _stop(args):
    """Stop the current running process of ATM."""
    pid_path = _get_pid_path(args.pid)
    process = _get_atm_process(pid_path)

    if process:
        process.terminate()

        for _ in range(args.timeout):
            if process.is_running():
                time.sleep(1)
            else:
                break

        if process.is_running():
            print('ATM was not able to stop after {} seconds.'.format(args.timeout))
            if args.force:
                print('Killing it.')
                process.kill()
                return True

            else:
                print('Use --force to kill it.')

        else:
            print('ATM stopped correctly.')
            return True

    else:
        print('ATM is not running.')


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

    # Worker
    worker = subparsers.add_parser('worker', parents=[parent])
    worker.set_defaults(action=_work)
    _add_common_arguments(worker)
    worker.add_argument('--cloud-mode', action='store_true', default=False,
                        help='Whether to run this worker in cloud mode')

    worker.add_argument('--dataruns', help='Only train on dataruns with these ids', nargs='+')
    worker.add_argument('--time', help='Number of seconds to run worker', type=int)

    worker.add_argument('--no-save', dest='save_files', action='store_false',
                        help="don't save models and metrics at all")

    # Server
    server = subparsers.add_parser('server', parents=[parent])
    server.set_defaults(action=_serve)
    _add_common_arguments(server)
    server.add_argument('--host', help='IP to listen at')
    server.add_argument('--port', help='Port to listen at', type=int)
    server.add_argument('--debug', action='store_true', help='Start the server in debug mode.')

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

    start.add_argument('--no-server', action='store_true', help='Do not start the REST server')
    start.add_argument('--host', help='IP to listen at')
    start.add_argument('--port', help='Port to listen at', type=int)
    start.add_argument('--pid', help='PID file to use.', default='atm.pid')
    start.add_argument('--debug', action='store_true', help='Start the server in debug mode.')

    # Status
    status = subparsers.add_parser('status', parents=[parent])
    status.set_defaults(action=_status)
    status.add_argument('--pid', help='PID file to use.', default='atm.pid')

    # restart
    restart = subparsers.add_parser('restart', parents=[parent])
    restart.set_defaults(action=_restart)
    _add_common_arguments(restart)
    restart.add_argument('--cloud-mode', action='store_true', default=False,
                         help='Whether to run this worker in cloud mode')
    restart.add_argument('--no-save', dest='save_files', default=True,
                         action='store_const', const=False,
                         help="don't save models and metrics at all")
    restart.add_argument('-w', '--workers', default=1, type=int, help='Number of workers')
    restart.add_argument('--no-server', action='store_true', help='Do not start the REST server')
    restart.add_argument('--host', help='IP to listen at')
    restart.add_argument('--port', help='Port to listen at', type=int)
    restart.add_argument('--pid', help='PID file to use.', default='atm.pid')
    restart.add_argument('--debug', action='store_true', help='restart the server in debug mode.')
    restart.add_argument('-t', '--timeout', default=5, type=int,
                         help='Seconds to wait before killing the process.')
    restart.add_argument('-f', '--force', action='store_true',
                         help='Kill the process if it does not terminate gracefully.')

    # Stop
    stop = subparsers.add_parser('stop', parents=[parent])
    stop.set_defaults(action=_stop)
    stop.add_argument('--pid', help='PID file to use.', default='atm.pid')
    stop.add_argument('-t', '--timeout', default=5, type=int,
                      help='Seconds to wait before killing the process.')
    stop.add_argument('-f', '--force', action='store_true',
                      help='Kill the process if it does not terminate gracefully.')

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

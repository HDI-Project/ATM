# -*- coding: utf-8 -*-

import argparse
import logging
import multiprocessing
import os
import time

import psutil
from daemon import DaemonContext
from lockfile.pidlockfile import PIDLockFile

from atm.api import create_app
from atm.config import AWSConfig, DatasetConfig, LogConfig, RunConfig, SQLConfig
from atm.core import ATM
from atm.data import copy_files, get_demos

LOGGER = logging.getLogger(__name__)


def _get_atm(args):
    sql_conf = SQLConfig(args)
    aws_conf = AWSConfig(args)
    log_conf = LogConfig(args)

    return ATM(**sql_conf.to_dict(), **aws_conf.to_dict(), **log_conf.to_dict())


def _work(args, wait=False):
    """Creates a single worker."""
    atm = _get_atm(args)

    atm.work(
        datarun_ids=getattr(args, 'dataruns', None),
        choose_randomly=False,
        save_files=args.save_files,
        cloud_mode=args.cloud_mode,
        total_time=getattr(args, 'total_time', None),
        wait=wait
    )


def _serve(args):
    """Launch the ATM API with the given host / port."""
    atm = _get_atm(args)
    app = create_app(atm, getattr(args, 'debug', False))
    app.run(host=args.host, port=args.port)


def _get_pid_path(pid):
    """Returns abspath of the pid file which is stored on the cwd."""
    pid_path = pid

    if not os.path.isabs(pid_path):
        pid_path = os.path.join(os.getcwd(), pid_path)

    return pid_path


def _get_atm_process(pid_path):
    """Return `psutil.Process` of the `pid` file. If the pidfile is stale it will release it."""
    pid_file = PIDLockFile(pid_path, timeout=1.0)

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
    """Launches the server/worker in daemon processes."""
    if args.server:
        LOGGER.info('Starting the REST API server')

        process = multiprocessing.Process(target=_serve, args=(args, ))
        process.daemon = True

        process.start()

    pool = multiprocessing.Pool(args.workers)
    for _ in range(args.workers):
        LOGGER.info('Starting background worker')
        pool.apply_async(_work, args=(args, True))

    pool.close()
    pool.join()


def _start(args):
    """Create a new process of ATM pointing the process to a certain `pid` file."""
    pid_path = _get_pid_path(args.pid)
    process = _get_atm_process(pid_path)

    if process:
        print('ATM is already running!')

    else:
        print('Starting ATM')

        if args.foreground:
            _start_background(args)

        else:
            pidfile = PIDLockFile(pid_path, timeout=1.0)

            with DaemonContext(pidfile=pidfile, working_directory=os.getcwd()):
                # Set up default log file if not already set
                if not args.logfile:
                    _logging_setup(args.verbose, 'atm.log')

                _start_background(args)


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

            else:
                print('Use --force to kill it.')

        else:
            print('ATM stopped correctly.')

    else:
        print('ATM is not running.')


def _restart(args):
    _stop(args)
    time.sleep(1)

    pid_path = _get_pid_path(args.pid)
    process = _get_atm_process(pid_path)

    if process:
        print('ATM did not stop correctly. Aborting')
    else:
        _start(args)


def _enter_data(args):
    atm = _get_atm(args)
    run_conf = RunConfig(args)
    dataset_conf = DatasetConfig(args)

    if run_conf.dataset_id is None:
        dataset = atm.add_dataset(**dataset_conf.to_dict())
        run_conf.dataset_id = dataset.id

    return atm.add_datarun(**run_conf.to_dict())


def _make_config(args):
    copy_files('*.yaml', ('config'))


def _get_demos(args):
    get_demos(args)


def _get_parser():
    logging_args = argparse.ArgumentParser(add_help=False)
    logging_args.add_argument('-v', '--verbose', action='count', default=0)
    logging_args.add_argument('-l', '--logfile')

    parser = argparse.ArgumentParser(description='ATM Command Line Interface',
                                     parents=[logging_args])

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    # Common Arguments
    sql_args = SQLConfig.get_parser()
    aws_args = AWSConfig.get_parser()
    log_args = LogConfig.get_parser()
    run_args = RunConfig.get_parser()
    dataset_args = DatasetConfig.get_parser()

    # Enter Data Parser
    enter_data_parents = [
        logging_args,
        sql_args,
        aws_args,
        dataset_args,
        log_args,
        run_args
    ]
    enter_data = subparsers.add_parser('enter_data', parents=enter_data_parents,
                                       help='Add a Dataset and trigger a Datarun on it.')
    enter_data.set_defaults(action=_enter_data)

    # Wroker Args
    worker_args = argparse.ArgumentParser(add_help=False)
    worker_args.add_argument('--cloud-mode', action='store_true', default=False,
                             help='Whether to run this worker in cloud mode')
    worker_args.add_argument('--no-save', dest='save_files', action='store_false',
                             help="don't save models and metrics at all")

    # Worker
    worker_parents = [
        logging_args,
        worker_args,
        sql_args,
        aws_args,
        log_args
    ]
    worker = subparsers.add_parser('worker', parents=worker_parents,
                                   help='Start a single worker in foreground.')
    worker.set_defaults(action=_work)
    worker.add_argument('--dataruns', help='Only train on dataruns with these ids', nargs='+')
    worker.add_argument('--total-time', help='Number of seconds to run worker', type=int)

    # Server Args
    server_args = argparse.ArgumentParser(add_help=False)
    server_args.add_argument('--host', help='IP to listen at')
    server_args.add_argument('--port', help='Port to listen at', type=int)

    # Server
    server = subparsers.add_parser('server', parents=[logging_args, server_args, sql_args],
                                   help='Start the REST API Server in foreground.')
    server.set_defaults(action=_serve)
    server.add_argument('--debug', help='Start in debug mode', action='store_true')
    # add_arguments_sql(server)

    # Background Args
    background_args = argparse.ArgumentParser(add_help=False)
    background_args.add_argument('--pid', help='PID file to use.', default='atm.pid')

    # Start Args
    start_args = argparse.ArgumentParser(add_help=False)
    start_args.add_argument('--foreground', action='store_true', help='Run on foreground')
    start_args.add_argument('-w', '--workers', default=1, type=int, help='Number of workers')
    start_args.add_argument('--no-server', dest='server', action='store_false',
                            help='Do not start the REST server')

    # Start
    start_parents = [
        logging_args,
        worker_args,
        server_args,
        background_args,
        start_args,
        sql_args,
        aws_args,
        log_args
    ]
    start = subparsers.add_parser('start', parents=start_parents,
                                  help='Start an ATM Local Cluster.')
    start.set_defaults(action=_start)

    # Status
    status = subparsers.add_parser('status', parents=[logging_args, background_args])
    status.set_defaults(action=_status)

    # Stop Args
    stop_args = argparse.ArgumentParser(add_help=False)
    stop_args.add_argument('-t', '--timeout', default=5, type=int,
                           help='Seconds to wait before killing the process.')
    stop_args.add_argument('-f', '--force', action='store_true',
                           help='Kill the process if it does not terminate gracefully.')

    # Stop
    stop = subparsers.add_parser('stop', parents=[logging_args, stop_args, background_args],
                                 help='Stop an ATM Local Cluster.')
    stop.set_defaults(action=_stop)

    # restart
    restart = subparsers.add_parser('restart', parents=start_parents + [stop_args],
                                    help='Restart an ATM Local Cluster.')
    restart.set_defaults(action=_restart)

    # Make Config
    make_config = subparsers.add_parser('make_config', parents=[logging_args],
                                        help='Generate a config templates folder in the cwd.')
    make_config.set_defaults(action=_make_config)

    # Get Demos
    get_demos = subparsers.add_parser('get_demos', parents=[logging_args],
                                      help='Create a demos folder and put the demo CSVs inside.')
    get_demos.set_defaults(action=_get_demos)

    return parser


def _logging_setup(verbosity=1, logfile=None):
    logger = logging.getLogger()
    log_level = (2 - verbosity) * 10
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

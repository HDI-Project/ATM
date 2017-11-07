#!/usr/bin/python2.7
import argparse
import os

from atm.enter_data import enter_data
from atm.worker import Worker
from atm.config import Config
from atm.utilities import download_file_s3

parser = argparse.ArgumentParser(description='''
Run a datarun with the specified tuner and selector.

To test custom btb implementations with this script, define a tuner class called
CustomTuner that inherits from a btb.tuning.Tuner and/or a selector called
CustomSelector that inherits from btb.selection.Selector. Pass the paths to both
of these files to --tuner and --selector, respectively.

The script will create a datarun using your tuner and selector, then run a
worker until the job is finished.
''')
parser.add_argument('--config', help='Location of config file', required=True)
parser.add_argument('--selector', help='path to BTB selector', default=None)
parser.add_argument('--tuner', help='path to BTB tuner', default=None)
parser.add_argument('--gridding', type=int, default=0,
                    help='if set, use gridding with this grid size')

args = parser.parse_args()


if __name__ == '__main__':
    print 'generating config...'
    config = Config(args.config)
    if args.tuner:
        config.set(Config.STRATEGY, Config.STRATEGY_SELECTION, args.tuner)
    if args.selector:
        config.set(Config.STRATEGY, Config.STRATEGY_FROZENS, args.selector)
    if args.gridding:
        # ALERT: you must set the value on the config as a string.
        # https://stackoverflow.com/questions/21484716/python-dictionary-from-config-parser-if-value-and-in-value-argument-of-typ
        config.set(Config.STRATEGY, Config.STRATEGY_GRIDDING,
                   str(args.gridding))

    print 'downloading data...'
    data_file = config.get(Config.DATA, Config.DATA_ALLDATAPATH)
    download_file_s3(config, data_file)

    print 'creating datarun...'
    datarun_id = enter_data(config)[0]

    print 'starting worker...'
    worker = Worker(config=config, datarun_id=datarun_id, save_files=False)
    worker.work()

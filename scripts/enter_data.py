from __future__ import print_function

import argparse
import warnings

from atm.config import (add_arguments_aws_s3, add_arguments_sql,
                        add_arguments_datarun, add_arguments_logging,
                        load_config, initialize_logging)
from atm.enter_data import enter_data

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
Creates a dataset (if necessary) and a datarun and adds them to the ModelHub.
All required arguments have default values. Running this script with no
arguments will create a new dataset with the file in data/pollution_1.csv and a
new datarun with the default arguments listed below.

You can pass yaml configuration files (--sql-config, --aws-config, --run-config)
instead of passing individual arguments. Any arguments in the config files will
override arguments passed on the command line. See the examples in the config/
folder for more information. """)
    # Add argparse arguments for aws, sql, and datarun config
    add_arguments_aws_s3(parser)
    add_arguments_sql(parser)
    add_arguments_datarun(parser)
    add_arguments_logging(parser)
    parser.add_argument('--run-per-partition', default=False, action='store_true',
                        help='if set, generate a new datarun for each hyperpartition')

    args = parser.parse_args()

    # create config objects from the config files and/or command line args
    sql_conf, run_conf, aws_conf, log_conf = load_config(sql_path=args.sql_config,
                                                         run_path=args.run_config,
                                                         aws_path=args.aws_config,
                                                         log_path=args.log_config,
                                                         **vars(args))
    initialize_logging(log_conf)

    # create and save the dataset and datarun
    enter_data(sql_conf, run_conf, aws_conf, args.run_per_partition)

import argparse
import os
from btb.config import Config
from btb.run import Run


def enter_data(config):
    data_filelist = config.get(Config.DATA, Config.DATA_FILELIST)
    alldatapath = config.get(Config.DATA, Config.DATA_ALLDATAPATH)
    trainpath = config.get(Config.DATA, Config.DATA_TRAINPATH)
    testpath = config.get(Config.DATA, Config.DATA_TESTPATH)
    dataset_description = config.get(Config.DATA, Config.DATA_DESCRIPTION)

    if alldatapath:
        isPreSplit = False
    else:
        isPreSplit = True

    if isPreSplit:
        runname = os.path.basename(trainpath)
        runname = runname.replace("_train", "")
        runname = runname.replace(".csv", "")
    else:
        runname = os.path.basename(alldatapath).replace(".csv", "")


    algorithm_codes = config.get(Config.RUN, Config.RUN_ALGORITHMS).split(', ')
    priority = int(config.get(Config.RUN, Config.RUN_PRIORITY))

    budget_type = config.get(Config.BUDGET, Config.BUDGET_TYPE)
    learner_budget = int(config.get(Config.BUDGET, Config.BUDGET_LEARNER))
    walltime_budget = int(config.get(Config.BUDGET, Config.BUDGET_WALLTIME))

    r_min = int(config.get(Config.STRATEGY, Config.STRATEGY_R))
    k_window = int(config.get(Config.STRATEGY, Config.STRATEGY_K))
    metric = config.get(Config.STRATEGY, Config.STRATEGY_METRIC)
    score_target = config.get(Config.STRATEGY, Config.STRATEGY_SCORE_TARGET)

    sample_selection = config.get(Config.STRATEGY, Config.STRATEGY_SELECTION)
    frozen_selection = config.get(Config.STRATEGY, Config.STRATEGY_FROZENS)

    description =  "__".join([sample_selection, frozen_selection])

    if (bool(dataset_description) and (len(dataset_description[0]) > 1000)):
        raise ValueError('Dataset description is more than 1000 characters.')

    if data_filelist:
        with open(data_filelist, 'r') as f:
            for line in f:
                alldatapath = line.strip()
                runname = os.path.basename(alldatapath).replace(".csv", "")

                Run(config, runname, description, metric, score_target,
                    sample_selection, frozen_selection, budget_type, priority,
                    k_window, r_min, algorithm_codes, learner_budget,
                    walltime_budget, alldatapath, dataset_description,
                    trainpath, testpath)
    else:
        Run(config, runname, description, metric, score_target,
            sample_selection, frozen_selection, budget_type, priority, k_window,
            r_min, algorithm_codes, learner_budget, walltime_budget,
            alldatapath, dataset_description, trainpath, testpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes a ')
    parser.add_argument('--configpath', help='Location of config file',
                        default=os.getenv('BTB_CONFIG_FILE', 'config/btb.cnf'),
                        required=False)
    args = parser.parse_args()
    config = Config(args.configpath)

    enter_data(config)

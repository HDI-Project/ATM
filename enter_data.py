import os
from atm.config import Config

# this has to go above Run import to get databases info loaded first
# TODO: config should only be loaded in one place
configpath = os.getenv('ATM_CONFIG_FILE', 'config/atm.cnf')
global config
config = Config(configpath)


from atm.run import Run


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

            Run(runname, description, metric, sample_selection, frozen_selection, budget_type, priority, k_window,
                r_min, algorithm_codes, learner_budget, walltime_budget, alldatapath, dataset_description, trainpath,
                testpath, configpath)
else:
    Run(runname, description, metric, sample_selection, frozen_selection, budget_type, priority, k_window, r_min,
        algorithm_codes, learner_budget, walltime_budget, alldatapath, dataset_description, trainpath, testpath,
        configpath)

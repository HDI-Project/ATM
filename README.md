BTB - Bayesian Tuning and Building
====

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://hdi-project.github.io/BTB/)

## Quick start guide

1. Install dependencies:
```
$ sudo apt install mysql-server mysql-client
$ virtualenv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

2. Set up MySQL (replace 'username' and 'password' with your choices):
```
$ mysql -u root -p            # you'll be prompted for your root password
mysql> GRANT ALL ON btb.\* TO 'username'@'localhost' IDENTIFIED BY 'password';
mysql> CREATE DATABASE btb;
mysql> exit
Bye
```

3. Set up the BTB database:
```
$ mysql -u username -p btb < setup/hyperbtb.sql
```
Check to make sure it worked.
```
$ mysql -u username -p btb
mysql> show tables;
+---------------+
| Tables_in_btb |
+---------------+
| algorithms    |
| dataruns      |
| frozen_sets   |
| learners      |
+---------------+
4 rows in set (0.00 sec)
``` 

4. Create a copy of the sample config file, and edit it to add your settings:
```
$ cp config/btb.cnf.template config/btb.cnf
$ vim config/btb.cnf
```
At the very least, you will need to change `alldatapath`, `models-dir`, and the
MySQL settings under `[datahub]`. This is the trickiest part; if you run into
issues later on, it's probably because of your config file. 

5. Create a datarun:
```
$ python btb/enter_data.py --configpath ./config/btb.cnf
```
You should get a lot of output, the end of which looks something like:

    ========== Summary ==========
    Algorithm classify_rf had 2 frozen sets
    Algorithm classify_gnb had 1 frozen sets
    Algorithm classify_dt had 2 frozen sets
    Algorithm classify_logreg had 6 frozen sets
    Sample selection: gp_ei
    Frozen selection: uniform
    <yourdataset:, frozen: uniform, sampling: gp_ei, budget: learner, status: pending>
    Datarun ID: 1
    LOCAL MODE: Train and test files only on local drive

The important piece of information is the datarun ID.

6. Start a worker, specifying your config file and the datarun you'd like to
   compute:
```
$ python btb/worker.py --configpath ./config/btb.cnf --datarun 1
```

This will start a process that computes learners and saves them in the models/
directory you configured. If you left the budget parameters in the config file
unchanged, it will compute 20 different learners before stopping. The output
should show which hyperparameters are being tested and the performance of each
learner (the "judgment metric"), plus the best overall performance so far.

    Classifier type: classify_logreg
    Params chosen:
            C = 8904.06127554
            _scale = True
            fit_intercept = False
            penalty = l2
            tol = 4.60893080631
            dual = True
            class_weight = auto

    Judgment metric (f1): 0.536 +- 0.067

And that's it! You can break out of the worker with Ctrl+C and restart it with
the same command; it will pick up right where it left off. You can also start
multiple workers at the same time in different terminals to parallelize the
work. When all 20 learners in your budget have been computed, all workers will
exit gracefully.

<!--Note: Any dataset with less than 30 samples will fail for the DBN classifier unless the DBN `minibatch_size` constant is changed to match the number of samples.-->

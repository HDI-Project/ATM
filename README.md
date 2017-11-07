ATM - Auto Tune Models
====

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://hdi-project.github.io/ATM/)

## Quick start setup
1. Clone project
```
$ git clone https://github.com/hdi-project/atm.git /path/to/atm
$ cd /path/to/atm
```

2. Install database
- for SQLite (simpler):
```
$ sudo apt install sqlite3
```

- for MySQL: 
```
$ sudo apt install mysql-server mysql-client
```

3. Install python dependencies
```
$ virtualenv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```
This will also install [btb](https://github.com/hdi-project/btb), another
project in development at DAI Lab, as an egg which will track changes to the git
repository.

4. Set up ATM database
- for SQLite:
```
$ sqlite3 atm.db < setup/sqlite-setup.sql
```

- for MySQL:
```
$ mysql -u username -p atm < setup/hyperatm.sql
```
Check to make sure it worked.
```
$ mysql -u username -p atm
mysql> show tables;
+---------------+
| Tables_in_atm |
+---------------+
| algorithms    |
| dataruns      |
| frozen_sets   |
| learners      |
+---------------+
4 rows in set (0.00 sec)
``` 

5. Create a copy of the sample config file, and edit it to add your settings:
```
$ cp config/atm.cnf.template config/atm.cnf
$ vim config/atm.cnf
```
At the very least, you will need to change `alldatapath`, `models-dir`, and the
database settings under `[datahub]`. This is the trickiest part; if you run into
issues later on, it's probably because of your config file. 

For SQLite, the [datahub] config should specify 'dialect' and 'database', and
leave everything else blank:

    dialect: sqlite
    database: ./atm.db
    username:
    password:
    host:
    port:
    query:

For MySQL, the config should look something like this: 

    dialect: mysql
    database: atm
    username: username
    password: password
    host: localhost
    port: 3306
    query:


6. Create a datarun
```
$ python atm/enter_data.py --configpath config/atm.cnf
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

7. Start a worker, specifying your config file and the datarun you'd like to
   compute on:
```
$ python atm/worker.py --configpath config/atm.cnf --datarun 1
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

## Testing Tuners and Selectors

The script `test_btb.py`, in the main directory, allows you to test different
BTB Tuners and Selectors using ATM. You will need AWS access keys from DAI lab
in order to download data from the S3 bucket. To use the script, edit your
config file as described above, then add the following fields (replacing the
API keys with your own):

```
[aws]
access_key: YOURACCESSKEY
secret_key: YoUrSECr3tKeY
s3_bucket: mit-dai-delphi-datastore
s3_folder: downloaded
```

Then, add the name of the data file you want to test:

```
[data]
alldatapath: filename.csv
```

To test a custom implementation of a BTB tuner or selector, define a new class called:
  * for Tuners, CustomTuner (inheriting from btb.tuning.Tuner)
  * for Selectors, CustomSelector (inheriting from btb.selection.Selector)
You can see examples of custom implementations in
btb/selection/custom\_selector.py and btb/tuning/custom\_tuning.py. Then, run
the script:

```
python test_btb.py --config config/atm.cnf --tuner /path/to/custom_tuner.py --selector /path/to/custom_selector.py
```

This will create a new datarun and start a worker to run it to completion. You
can also choose to use the default tuners and selectors included with BTB:

```
python test_btb.py --config config/atm.cnf --tuner gp --selector ucb1
```

<!--Note: Any dataset with less than 30 samples will fail for the DBN classifier unless the DBN `minibatch_size` constant is changed to match the number of samples.-->

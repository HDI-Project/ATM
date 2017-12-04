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

4. (Optional) Create copies of the sample configuration files, and edit them to
   add your settings. 
```
$ cp config/templates/*.yaml config/
$ vim config/*.yaml
```
If you want, you can skip this step and specify all custom configurations on the
command line later.

In `run_config.yaml`, you will need to modify `train_path` to use a new dataset.

If you are using a MySQL database, you will need to change the configuration in
`sql_config.yaml`; the default settings will create a new SQLite database called
atm.db.

For SQLite, the [datahub] config should specify 'dialect' and 'database', and
leave everything else blank:

    dialect: sqlite
    database: atm.db
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

If you need to download data from an Amazon S3 bucket, you should update
`aws_config.yaml` with your credentials.

5. Create a datarun
```
$ python atm/enter_data.py --command --line --args
```

This command will create a Datarun and a Dataset (if necessary) and store both
in the ModelHub database. If you have specified everything in .yaml config
files, you should call it like this:

```
$ python atm/enter_data.py --sql-config config/sql_config.yaml \
  --aws-config config/aws_config.yaml \
  --run-config config/run_config.yaml
```

Otherwise, the script will use the default configuration values (specified in
atm/config.py), except where overridden by command line arguments. You can also
use a mixture of config files and command line args; any command line arguments
you specify will override the values found in config files.

The command should produce a lot of output, the end of which looks something
like this:

    ========== Summary ==========
    Training data: data/test/pollution_1.csv
    Test data: <None>
    Dataset ID: 1
    Frozen set selection strategy: uniform
    Parameter tuning strategy: gp_ei
    Budget: 100 (learner)
    Datarun ID: 1

The most important piece of information is the datarun ID.

6. Start a worker, specifying your config files and the datarun(s) you'd like to
   compute on:
```
$ python atm/worker.py --sql-config config/sql_config.yaml \
  --aws-config config/aws_config.yaml --dataruns 1
```

This will start a process that computes learners and saves them to the model
directory you configured. Again, you can run worker.py without any arguments,
and the default configuration values will be used. If you don't specify any
dataruns, the worker will periodically check the ModelHub database for new
dataruns, and compute learners for any it finds in order of priority.  The
output should show which hyperparameters are being tested and the performance of
each learner (the "judgment metric"), plus the best overall performance so far.

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
    Best so far (learner 21): 0.716 +- 0.035

And that's it! You can break out of the worker with Ctrl+C and restart it with
the same command; it will pick up right where it left off. You can also start
multiple workers at the same time in different terminals to parallelize the
work. When all 100 learners in your budget have been computed, all workers will
exit gracefully.

<!--## Testing Tuners and Selectors-->

<!--The script `test_btb.py`, in the main directory, allows you to test different-->
<!--BTB Tuners and Selectors using ATM. You will need AWS access keys from DAI lab-->
<!--in order to download data from the S3 bucket. To use the script, -->
<!--config file as described above, then add the following fields (replacing the-->
<!--API keys with your own):-->

<!--```-->
<!--[aws]-->
<!--access_key: YOURACCESSKEY-->
<!--secret_key: YoUrSECr3tKeY-->
<!--s3_bucket: mit-dai-delphi-datastore-->
<!--s3_folder: downloaded-->
<!--```-->

<!--Then, add the name of the data file you want to test:-->

<!--```-->
<!--[data]-->
<!--alldatapath: filename.csv-->
<!--```-->

<!--To test a custom implementation of a BTB tuner or selector, define a new class called:-->
  <!--* for Tuners, CustomTuner (inheriting from btb.tuning.Tuner)-->
  <!--* for Selectors, CustomSelector (inheriting from btb.selection.Selector)-->
<!--You can see examples of custom implementations in-->
<!--btb/selection/custom\_selector.py and btb/tuning/custom\_tuning.py. Then, run-->
<!--the script:-->

<!--```-->
<!--python test_btb.py --config config/atm.cnf --tuner /path/to/custom_tuner.py --selector /path/to/custom_selector.py-->
<!--```-->

<!--This will create a new datarun and start a worker to run it to completion. You-->
<!--can also choose to use the default tuners and selectors included with BTB:-->

<!--```-->
<!--python test_btb.py --config config/atm.cnf --tuner gp --selector ucb1-->
<!--```-->

<!--Note: Any dataset with less than 30 samples will fail for the DBN classifier unless the DBN `minibatch_size` constant is changed to match the number of samples.-->

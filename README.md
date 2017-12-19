ATM - Auto Tune Models
====
ATM is an open source software library under ["The human data interaction project"](https://hdi-dai.lids.mit.edu/) at MIT.  It is a distributed scalable AutoML system designed with ease of use in mind. ATM takes in data with pre-extracted feature vectors and labels (target column) in a simple CSV file format. It attempts to learn several classifiers (machine learning models to predict the label) in parallel. In the end, ATM returns a number of classifiers and the best classifier with a specified set of hyperparameters. 

## Current status
atm and the accompanying library btb are under active development (transitioning from an older system to new). In the next couple of weeks we intend to update its documentation, its testing infrastructure, provide apis and establish a framework for the community to contribute. Stay tuned for updates. Meanwhile, if you have any questions, or if would like to receive updates: **please email to dailabmit@gmail.com. **

## Quick start setup
Below we will give a quick tutorial of how to run atm on your desktop. We will use a featurized dataset, loaded in ``

1. **Clone project**.
   ```
      $ git clone https://github.com/hdi-project/atm.git /path/to/atm
      $ cd /path/to/atm
   ```

2. **Install database.**
   - for SQLite (simpler):
   ```
      $ sudo apt install sqlite3
   ```

   - for MySQL: 
   ```
      $ sudo apt install mysql-server mysql-client
   ```

3. **Install python dependencies.**
   ```
      $ virtualenv venv
      $ . venv/bin/activate
      $ pip install -r requirements.txt
   ```
   This will also install [btb](https://github.com/hdi-project/btb), another
   project in development at DAI Lab, as an egg which will track changes to the git
   repository.


4. **(Optional) Create copies of the sample configuration files, and edit them to
   add your settings.** 

      Saving configuration as YAML files is an easy way to save complicated setups or
      share them with team members. However, if you want, you can skip this step and
      specify all configuration parameters on the command line later.

      If you do want to use YAML config files, you should start with the templates
      provided in `config/templates` and modify them to suit your own needs.
      ```
         $ cp config/templates/*.yaml config/
         $ vim config/*.yaml
      ```

   `run_config.yaml` contains all the settings for a single Dataset and Datarun.
   You will need to modify `train_path` at the very least in order to use your own
   dataset.

   `sql_config.yaml` contains the settings for the ModelHub SQL database. The
   default configuration will connect to (and create if necessary) a SQLite
   database called atm.db. If you are using a MySQL database, you will need to
   change it to something like this: 
   ```
      dialect: mysql
      database: atm
      username: username
      password: password
      host: localhost
      port: 3306
      query:
    ```

   If you need to download data from an Amazon S3 bucket, you should update
   `aws_config.yaml` with your credentials.

5. Create a datarun.
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
   ```
      ========== Summary ==========
      Training data: data/test/pollution_1.csv
      Test data: <None>
      Dataset ID: 1
      Frozen set selection strategy: uniform
      Parameter tuning strategy: gp_ei
      Budget: 100 (learner)
      Datarun ID: 1
   ```

   The most important piece of information is the datarun ID.

6. **Start a worker, specifying your config files and the datarun(s) you'd like to
   compute on.**
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
   ```
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
   ```
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

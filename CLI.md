# Command Line Interface

**ATM** provides a simple command line client that will allow you to run ATM directly
from your terminal by simply passing it the path to a CSV file.

## Quickstart

In this example, we will use the default values that are provided in the code in order to generate
classifiers.

### 1. Get the demo data

The first step in order to run **ATM** is to obtain the demo datasets that will be used in during
the rest of the tutorial.

For this demo we will be using the pollution csv from the
[demos bucket](https://atm-data.s3.amazonaws.com/index.html), which you can download from
[here](https://atm-data.s3.amazonaws.com/pollution_1.csv).


### 2. Create a dataset and generate it's dataruns

Once you have obtained your demo dataset, now it's time to create a `dataset` object inside the
database. Our command line also triggers the generation of `datarun` objects for this dataset in
order to automate this process as much as possible:

```bash
atm enter_data --train-path path/to/pollution_1.csv
```

Bear in mind that `--train-path` argument can be a local path, an URL link to the CSV file or an
complete S3 Bucket path.

If you run this command, you will create a dataset with the default values, which is using the
`pollution_1.csv` dataset from the demo datasets.

A print, with similar information to this, should be printed:

```bash
method logreg has 6 hyperpartitions
method dt has 2 hyperpartitions
method knn has 24 hyperpartitions
Dataruns created. Summary:
	Dataset ID: 1
	Training data: path/to/pollution_1.csv
	Test data: None
	Datarun ID: 1
	Hyperpartition selection strategy: uniform
	Parameter tuning strategy: uniform
	Budget: 100 (classifier)
```

For more information about the arguments that this command line accepts, please run:

```bash
atm enter_data --help
```

### 3. Start a worker

**ATM** requieres a worker to process the dataruns that are not completed and stored inside the
database. This worker process will be runing until there are no dataruns `pending`.

In order to launch such a process, execute:

```bash
atm worker
```

This will start a process that builds classifiers, tests them, and saves them to the `./models/`
directory. The output should show which hyperparameters are being tested and the performance of
each classifier (the "judgment metric"), plus the best overall performance so far.

Prints similar to this one will apear repeatedly on your console while the `worker` is processing
the datarun:

```bash
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
Best so far (classifier 21): 0.716 +- 0.035
```

Occasionally, a worker will encounter an error in the process of building and testing a
classifier. When this happens, the worker will print error data to the console, log the error in
the database, and move on to the next classifier.

You can break out of the worker with <kbd>Ctrl</kbd>+<kbd>c</kbd> and restart it with the same
command; it will pick up right where it left off. You can also run the command simultaneously in
different terminals to parallelize the work -- all workers will refer to the same ModelHub
database. When all 100 classifiers in your budget have been built, all workers will exit gracefully.

This command aswell offers more information about the arguments that this command line accepts:

```
atm worker --help
```

## Command Line Arguments

You can specify each argument individually on the command line. The names of the
variables are the same as those described [here](https://hdi-project.github.io/ATM/configuring_atm.html#arguments).
SQL configuration variables must be prepended by `sql-`, and AWS config variables must be
prepended by `aws-`.

### Using command line arguments

Using command line arguments is convenient for quick experiments, or for cases where you
need to change just a couple of values from the default configuration. For example:

```bash
atm enter_data --train-path ./data/my-custom-data.csv \
              --test-path ./data/my-custom-test-data.csv \
              --selector bestkvel
```

You can also use a mixture of config files and command line arguments; any command line
arguments you specify will override the values found in config files.

### Using YAML configuration files

You can also save the configuration as YAML files is an easy way to save complicated setups
or share them with team members.

You should start with the templates provided by the `atm make_config` command:

```bash
atm make_config
```

This will generate a folder called `config/templates` in your current working directory which
will contain 5 files, which you will need to copy over to the `config` folder and edit according
to your needs:

```bash
cp config/templates/*.yaml config/
vim config/*.yaml
```

`run.yaml` contains all the settings for a single dataset and datarun. Specify the `train_path`
to point to your own dataset.

`sql.yaml` contains the settings for the ModelHub SQL database. The default configuration will
connect to (and create if necessary) a SQLite database at `./atm.db` relative to the directory
from which `enter_data.py` is run. If you are using a MySQL database, you will need to change
the file to something like this:

```
dialect: mysql
database: atm
username: username
password: password
host: localhost
port: 3306
query:
```

`aws.yaml` should contain the settings for running ATM in the cloud. This is not necessary
for local operation.

Once your YAML files have been updated, run the datarun creation command and pass it the paths
to your new config files:

```bash
atm enter_data --sql-config config/sql.yaml \
              --aws-config config/aws.yaml \
              --run-config config/run.yaml
```

It's important that the SQL configuration used by the worker matches the configuration you
passed to `enter_data` -- otherwise, the worker will be looking in the wrong ModelHub
database for its datarun!

```
atm worker --sql-config config/sql.yaml \
          --aws-config config/aws.yaml \
```

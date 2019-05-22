# Using ATM with your own data

If you want to use the system for your own dataset, convert your data to a CSV file with the
following format:

* Each column is a feature (or the label).
* Each row is a training example.
* The first row is the header row, which contains names for each column of data.
* A single column (the *target* or *label*), if this columns name is different than `class` you
will have to provide it.


# Using custom configuration:

## Python

If you would like to create a custom instance of **ATM**  you can specify this with arguments
during instantiation.

### Using a config.yaml file

You can create and provide a path to a `config.yaml` file that contains all the configuration
for your **ATM** instance.

To create such a `yaml` file ,execute the following command that will generate the config tempalte
files:

```bash
atm make_config
```

This will create a folder named `config` with the following structure:

```bash
config
└── templates
    ├── aws.yaml
    ├── config.yaml
    ├── log-script.yaml
    ├── log.yaml
    ├── run.yaml
    └── sql.yaml
```

There you will find the `config.yaml` template that you can copy and modify setting the arguments
that you would like to use.

```
cp config/templates/config.yaml config/
vim config/config.yaml
```

An example, using mysql, would be:

```
dialect: mysql
database: atm
username: username
password: password
host: localhost
port: 3306
```

Then you can simply instantiate `ATM` giving it the path to this `config.yaml`:

```
from atm import ATM

atm = ATM(config_file='config/config.yaml')
```

### Using arguments

The ATM initiation accepts the same arguments aswell:

```python
from atm import ATM

atm = ATM(
    dialect='mysql',
    database='atm',
    username='username',
    password='password',
    host='localhost',
    port=3306
)
```

This will create the same `ATM` as the one with the `config.yaml`.

### Using the `run` method with your own data

**ATM** `run` method allows you to specify different arguments which have default values, however,
you may need to change some of them in order to make it work with your dataset.
For example, if the column target that you are trying to predict is not `class` then you will
have to specify the name of it.

Here is a list of the arguments that `run` method accepts:

* `class_column`: The target column that has to be classified. By default it is set to `class`,
type str.

* `train_path`: Path to a `csv` file that follows the data specifications. This must be specified,
type `str`.

* `test_path`: Path to a `csv` file that follows the data specifications, this one will be used
as test data, if this is `None`, **ATM**, will split the train data into train and test. Type `str`.

* `budget`: Amount of `classifiers` or amount of `minutes` to run, type `int`.
* `budget_type`: Type of the `budget`, by default it's `classifier`, can be changed to `walltime`,
type `str`
* `gridding`: Gridding factor, by default set to `0` which means that no gridding will be
performed, type `int`.

* `k_window`: Number of previous scores considered by `k selector` methods. Default is `3`,
type `int`
* `methods`: Method or a list of methods to use for classification. Each method can either be one
of the pre-defined method codes listed below or a path to a JSON file defining a custom method.
Default is `['logreg', 'dt', 'knn']`, type is `str` or a `list` like. A complete list of the
default choices in **ATM** are:

    * logreg
    * svm
    * sgd
    * dt
    * et
    * rf
    * gnb
    * mnb
    * bnb
    * gp
    * pa
    * knn
    * mlp
    * ada

* `metric`: Metric by which **ATM** should evaluate the classifiers. The metric function specified
here will be used to compute the judgment metric for each classifier. Default `metric` is set to
`f1`, type `str`. The rest of metrics that we support at the moment is as follows:

    * roc_auc_micro
    * rank_accuracy
    * f1_micro
    * accuracy
    * roc_auc_macro
    * ap
    * cohen_kappa
    * f1
    * f1_macro
    * mcc


* `r_minimum`:  Number of random runs to perform before tuning can occur. Default value is `2`,
type `int`.
* `run_per_partition`: If true, generate a new datarun for each hyperpartition. Default is `False`,
type `bool`.
* `score_target`: Determines which judgment metric will be used to search the hyperparameter space.
`cv` will use the mean cross-validated performance, `test` will use the performance on a test
dataset, and `mu_sigma` will use the lower confidence bound on the CV performance. Default is `cv`,
type `str`.

* `selector`: Type of [BTB](https://github.com/HDI-Project/BTB/) selector to use. A list of them at
the moment is `[uniform, ucb1, bestk, bestkvel, purebestkvel, recentk, hieralg]`. Default is set to
`uniform`, type `str`.

* `tuner`: Type of [BTB](https://github.com/HDI-Project/BTB/) tuner to use. A list of them at the
moment is `[uniform, gp, gp_ei, gp_eivel]`. Default is set to `uniform`, type `str`.

## Command Line

### Using command line arguments

You can specify each argument individually on the command line. The names of the
variables are the same as those in the YAML files. SQL configuration variables must be
prepended by `sql-`, and AWS config variables must be prepended by `aws-`.

Using command line arguments is convenient for quick experiments, or for cases where you
need to change just a couple of values from the default configuration. For example:

```
atm enter_data --train-path ./data/my-custom-data.csv --selector bestkvel
```

You can also use a mixture of config files and command line arguments; any command line
arguments you specify will override the values found in config files.

### Using YAML configuration files

You can also save the configuration as YAML files is an easy way to save complicated setups
or share them with team members.

You should start with the templates provided by the `atm make_config` command:

```
atm make_config
```

This will generate a folder called `config/templates` in your current working directory which
will contain 5 files, which you will need to copy over to the `config` folder and edit according
to your needs:

```
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

```
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

## Setting up a distributed Database

ATM uses a database to store information about datasets, dataruns and classifiers.
It's currently compatible with the SQLite3 and MySQL dialects.

For first-time and casual users, the SQLite3 is used by default without any required
step from the user.

However, if you're planning on running large, distributed, or performance-intensive jobs,
you might prefer using MySQL.

If you do not have a MySQL database already prepared, you can follow the next steps in order
install it and parepare it for ATM:


### 1. Install mysql-server

```bash
sudo apt-get install mysql-server
```

In the latest versions of MySQL no input for the user is required for this step, but
in older versions the installation process will require the user to input a password
for the MySQL root user.

If this happens, keep track of the password that you set, as you will need it in the
next step.

### 2. Log into your MySQL instance as root

If no password was required during the installation of MySQL, you should be able to
log in with the following command.

```bash
sudo mysql
```

If a MySQL Root password was required, you will need to execute this other command:

```bash
sudo mysql -u root -p
```

and input the password that you used during the installation when prompted.

### 3. Create a new Database for ATM

Once you are logged in, execute the following three commands to create a database
called `atm` and a user also called `atm` with write permissions on it:

```bash
$ mysql> CREATE DATABASE atm;
$ mysql> CREATE USER 'atm'@'localhost' IDENTIFIED BY 'set-your-own-password-here';
$ mysql> GRANT ALL PRIVILEGES ON atm.* TO 'atm'@'localhost';
```

### 4. Test your settings

After you have executed the previous three commands and exited the mysql prompt,
you can test your settings by executing the following command and inputing the
password that you used in the previous step when prompted:

```bash
mysql -u atm -p
```

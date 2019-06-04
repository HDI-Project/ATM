# Configuring ATM

Nearly every part of ATM is configurable. For example, you can specify which machine-learning
algorithms ATM should try, which metrics it computes (such as F1 score and ROC/AUC), and which
method it uses to search through the space of hyperparameters (using another HDI Project library,
BTB). You can also constrain ATM to find the best model within a limited amount of time or by
training a limited amount of total models.

## Arguments

The `atm.ATM` class accepts a series of arguments to configure the environment where
ATM is run.

### Database Configuration

These arguments specify the database configuration, including which type of database is being
used and how to connect to it.

The arguments for **SQL** are:

* `dialect`: type of the sql database. Choices are sqlite or mysql. The default is `sqlite`.
* `database`: name or path of the database. The default is `atm.db`.
* `username`: username for the database to be used. The default is `None`.
* `password`: password for the username. The default is `None`.
* `host`: IP adress or 'localhost' to where the connection is going to be established. The
default is `None`.
* `port`: Port number of where the database is listening, default is `None`.
* `query`: additional query to be executed for the login process, default is `None`.

An example of creating an instance with `mysql` database:

```python
from atm import ATM

atm = ATM(
    dialect='mysql',
    database='atm',
    username='admin',
    password='password',
    host='localhost',
    port=3306
)
```

### AWS

The following arguments specify the [AWS](https://aws.amazon.com/) configuration. Bear in mind that
you can have the `access_key` and `secret_key` already configured on your machine if you follow
the steps [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#configuring-credentials).
[Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) will use
them by default, however if you specify them during instantiation, this will be the ones used.

* `access_key`: aws access key id provided from amazon.
* `secret_key`: aws secret key provided from amazon.
* `s3_bucket`: S3 bucket to be used to store the models and metrics.
* `s3_folder`: folder inside the bucket where the models and metrics will be saved.

**Note** all this arguments are `None` by default, and they should be passed as a `string`.


An exmaple of creating an instance with `aws` configuration is:

```python
from atm import ATM

atm = ATM(
    access_key='my_aws_key_id',
    secret_key='my_aws_secret_key',
    s3_bucket='my_bucket',
    s3_folder='my_folder'
)
```

### Log

The following arguments specify the configuration to where the models and metrics will be stored
and if we would like a verbose version of the metrics.

* `models_dir`: local folder where the models should be saved, default is `models`.
* `metrics_dir`: local folder where the models should be saved, default is `metrics`.
* `verbose_metrics`: whether or not to store verbose metrics, default is `False`.

An example of creating an instance with `log` configuration is:

```python
from atm import ATM

atm = ATM(
    models_dir='my_path_to_models',
    metrics_dir='my_path_to_metrics',
    verbose_metrics=True
)
```

### Dataset

The following arguments are used to specify the `dataset` creation inside the database.

* `train_path`: local path, URL or S3 bucket URL, to a CSV file that follows the
[Data Format](https://hdi-project.github.io/ATM/#data-format) and specifies the traininig data
for the models.

* `test_path`: local path, URL or S3 bucket URL, to a CSV file that follows the
[Data Format](https://hdi-project.github.io/ATM/#data-format) and specifies the test data for
the models, if this is `None` the training data will be splited in train and test.

* `name`: a name for the `dataset`, if it's not set an `md5` will be generated from the path.
* `description`: short description about the dataset, default is `None`.
* `class_column`: name of the column that is being the target of our predictions,
default is `class`.


An example of using this arguments in our `atm.run` method is:

```python
from atm import ATM

atm = ATM()

results = atm.run(
    train_path='path/to/train.csv',
    test_path='path/to/test.csv',
    name='test',
    description='Test data',
    class_column='test_column'
)
```

### Datarun

The following arguments are used to specify the `datarun` creation inside the database. This
configuration it's important for the behaviour and metrics of our `classifiers`.

* `budget`: amount of `classifiers` or amount of `minutes` to run, type `int`.

* `budget_type`: Type of the `budget`, by default it's `classifier`, can be changed to `walltime`,
type `str`.

* `gridding`: Gridding factor, by default set to `0` which means that no gridding will be
performed, type `int`.

* `k_window`: Number of previous scores considered by `k selector` methods. Default is `3`,
type `int`.

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

    * `roc_auc_micro`
    * `rank_accuracy`
    * `f1_micro`
    * `accuracy`
    * `roc_auc_macro`
    * `ap`
    * `cohen_kappa`
    * `f1`
    * `f1_macro`
    * `mcc`

* `r_minimum`:  Number of random runs to perform before tuning can occur. Default value is `2`,
type `int`.

* `run_per_partition`: If true, generate a new datarun for each hyperpartition. Default is
`False`, type `bool`.

* `score_target`: Determines which judgment metric will be used to search the hyperparameter
space. `cv` will use the mean cross-validated performance, `test` will use the performance on a
test dataset, and `mu_sigma` will use the lower confidence bound on the CV performance. Default is
`cv`, type `str`.

* `priority`: the priority for this datarun, the higher value is the most important.

* `selector`: Type of [BTB](https://github.com/HDI-Project/BTB/) selector to use. A list of them
at the moment is `[uniform, ucb1, bestk, bestkvel, purebestkvel, recentk, hieralg]`. Default is
set to `uniform`, type `str`.

* `tuner`: Type of [BTB](https://github.com/HDI-Project/BTB/) tuner to use. A list of them at the
moment is `[uniform, gp, gp_ei, gp_eivel]`. Default is set to `uniform`, type `str`.

An example using `atm.run` method with this arguments is:

```python
from atm import ATM

atm = ATM()

results = atm.run(
    budget=200,
    budget_type='classifier',
    gridding=0,
    k_window=4,
    metric='f1_macro',
    methods=['logreg']
    r_minimum=2,
    run_per_partition=True,
    score_target='cv',
    priority=9,
    selector='uniform',
    tuner='uniform',
    deadline=None,
)
```

## Using ATM with your own data

If you would like to use the system for your own dataset, convert your data to a csv file with
the specified [data format](https://hdi-project.github.io/ATM/#data-format).

Once having your dataset ready to use, you will simply have to provide the path to this CSV in one
of the supported formats (local path, URL, or a complete AWS S3 Bucket path). **Bear in mind** that
if you specify an S3 Bucket path, the propper access keys should be configured.


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

<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“ATM” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>


[![CircleCI][circleci-img]][circleci-url]
[![Travis][travis-img]][travis-url]
[![Version][pypi-img]][pypi-url]
[![Coverage Status][codecov-img]][codecov-url]

[circleci-img]: https://circleci.com/gh/HDI-Project/ATM.svg?style=shield
[circleci-url]: https://circleci.com/gh/HDI-Project/ATM
[travis-img]: https://travis-ci.org/HDI-Project/ATM.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/ATM
[pypi-img]: https://img.shields.io/pypi/v/atm.svg
[pypi-url]: https://pypi.python.org/pypi/atm
[codecov-img]: https://codecov.io/gh/HDI-project/ATM/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/HDI-project/ATM


# ATM - Auto Tune Models

- Free software: MIT license
- Documentation: https://hdi-project.github.io/ATM/

ATM is an open source software library under the
[*Human Data Interaction* project](https://hdi-dai.lids.mit.edu/)
(HDI) at MIT. It is a distributed, scalable AutoML system designed with ease of use in mind.

## Summary

For a given classification problem, ATM's goal is to find

1. a classification *method*, such as *decision tree*, *support vector machine*,
or *random forest*, and
2. a set of *hyperparameters* for that method

which generate the best classifier possible.

ATM takes in a dataset with pre-extracted feature vectors and labels as a CSV file.
It then begins training and testing classifiers (machine learning models) in parallel.
As time goes on, ATM will use the results of previous classifiers to intelligently select
which methods and hyperparameters to try next.
Along the way, ATM saves data about each classifier it trains, including the hyperparameters
used to train it, extensive performance metrics, and a serialized version of the model itself.

ATM has the following features:

* It allows users to run the system for multiple datasets and multiple
problem configurations in parallel.
* It can be run locally, on AWS\*, or on a custom compute cluster\*
* It can be configured to use a variety of AutoML approaches for hyperparameter tuning and
selection, available in the accompanying library [btb](https://github.com/hdi-project/btb)
* It stores models, metrics and cross validated accuracy information about each
classifier it has trained.

\**work in progress! See issue [#40](https://github.com/HDI-Project/ATM/issues/40)*

## Current status

ATM and the accompanying library BTB are under active development.
We have made the transition and improvements.

## Setup

This section describes the quickest way to get started with ATM on a machine running Ubuntu Linux.
We hope to have more in-depth guides in the future, but for now, you should be able to substitute
commands for the package manager of your choice to get ATM up and running on most modern
Unix-based systems.

### Requirements

ATM is compatible with and has been tested on Python 2.7, 3.5, and 3.6.

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
where **ATM** is run.

### Installation

To get started with **ATM**, we recommend using [pip](https://pip.pypa.io/en/stable/).

Once you have created and activated your virtualenv, execute:

```bash
pip install atm
```

Alternatively, you can clone the repository and install it from source by running
`make install`:

```bash
git clone git@github.com:HDI-Project/ATM.git
cd ATM
make install
```

For development, you can use the `make install-develop` command instead in order to install all
the required dependencies for testing and linting.


## Quick Usage

Below we will give a quick tutorial of how to run ATM on your desktop.
We will use a featurized dataset, named `pollution_1.csv`, reffer to section `0` to generate it.
This is one of the datasets available on [openml.org](https://www.openml.org).
More details can be found [here](https://www.openml.org/d/542).
In this problem the goal is predict `mortality` using the metrics associated with the air
pollution. Below we show a snapshot of the `csv` file.
The data has 15 features and the last column is the `class` label.


|PREC  |JANT  |JULT  |OVR65 |POPN  |EDUC  |HOUS  |DENS  |NONW  |WWDRK |POOR  |HC    |NOX   |SO@   |HUMID |class |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|35    |23    |72    |11.1  |3.14  |11    |78.8  |4281  |3.5   |50.7  |14.4  |8     |10    |39    |57    |1     |
|44    |29    |74    |10.4  |3.21  |9.8   |81.6  |4260  |0.8   |39.4  |12.4  |6     |6     |33    |54    |1     |
|47    |45    |79    |6.5   |3.41  |11.1  |77.5  |3125  |27.1  |50.2  |20.6  |18    |8     |24    |56    |1     |
|43    |35    |77    |7.6   |3.44  |9.6   |84.6  |6441  |24.4  |43.7  |14.3  |43    |38    |206   |55    |1     |
|53    |45    |80    |7.7   |3.45  |10.2  |66.8  |3325  |38.5  |43.1  |25.5  |30    |32    |72    |54    |1     |
|43    |30    |74    |10.9  |3.23  |12.1  |83.9  |4679  |3.5   |49.2  |11.3  |21    |32    |62    |56    |0     |
|45    |30    |73    |9.3   |3.29  |10.6  |86    |2140  |5.3   |40.4  |10.5  |6     |4     |4     |56    |0     |
|..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |
|..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |
|..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |..    |
|37    |31    |75    |8     |3.26  |11.9  |78.4  |4259  |13.1  |49.6  |13.9  |23    |9     |15    |58    |1     |
|35    |46    |85    |7.1   |3.22  |11.8  |79.9  |1441  |14.8  |51.2  |16.1  |1     |1     |1     |54    |0     |


0. **Generate the demo datasets**

    In order to follow this guide, you will have to get the demo datasets that we provide with this
    package.

    To do so, we provide you with a simple command:

    ```bash
    atm get_demos
    ```

    The command generates four datasets in the current directory inside a folder `demos`.
    ```bash
    Generating file demos/pollution_1.csv
    Generating file demos/iris.data.csv
    Generating file demos/pitchfork_genres.csv
    Generating file demos/adult_data_300.csv
    ```


1. **Create a datarun**

   ```
   atm enter_data
   ```

   This command will create a `datarun`. In ATM, a "datarun" is a single logical machine learning
   task. If you run the above command without any arguments, it will use the default settings
   found in the code to create a new SQLite3 database at `./atm.db`, create a new
   `dataset` instance which refers to the data above, and create a `datarun` instance which
   points to that dataset. More about what is stored in this database and what is it used for
   can be found [here](https://cyphe.rs/static/atm.pdf).

   The command should produce a lot of output, the end of which looks something like this:

   ```
   ========== Summary ==========
   Training data: data/test/pollution_1.csv
   Test data: <None>
   Dataset ID: 1
   Frozen set selection strategy: uniform
   Parameter tuning strategy: gp_ei
   Budget: 100 (classifier)
   Datarun ID: 1
   ```

   The most important piece of information is the datarun ID.


2. **Start a worker**

   ```
   atm worker
   ```

   This will start a process that builds classifiers, tests them, and saves them to the
   `./models/` directory. The output should show which hyperparameters are being tested
   and the performance of each classifier (the "judgment metric"), plus the best overall
   performance so far.

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
   Best so far (classifier 21): 0.716 +- 0.035
   ```

   Occasionally, a worker will encounter an error in the process of building and testing a
   classifier. When this happens, the worker will print error data to the terminal, log the
   error in the database, and move on to the next classifier.

And that's it! You can break out of the worker with <kbd>Ctrl</kbd>+<kbd>c</kbd> and restart
it with the same command; it will pick up right where it left off. You can also run the
command simultaneously in different terminals to parallelize the work -- all workers will
refer to the same ModelHub database. When all 100 classifiers in your budget have been built,
all workers will exit gracefully.


### Using ATM with Python

0. **Generate the demo datasets**

    If you have not generated the demo datasets, you can do so by calling `get_demos` method from
    the `data` module without arguments:

    ```python
    from atm import data

    demo_datasets = data.get_demos()
    ```

    The method `get_demos` will print and also return a dictionary where the files have been
    generated.


1. **Auto Tune Models over a CSV file**

    In order to Auto Tume Models over a csv file, we first have to create a instance of `ATM`.

    ```python
    from atm import ATM

    atm = ATM()
    ```

    This will create an instance with the default settings for `ATM`.

    Once you have the instance ready, you can use the method `run`, by setting the argument
    `train_path` to the csv training path. **Note** we will use the `dictionary` generated before
    to get the path of the `pollution_1.csv`.

    ```python
    path = demo_datasets.get('pollution_1.csv')

    results = atm.run(train_path=path)
    ```

    This process will display a progress bar during it's execution, on your python interpreter, you
    will be able to see something similar to this:

    ```python
    Processing dataset demos/pollution_1.csv
    100%|##########################| 100/100 [00:10<00:00,  6.09it/s]Classifier budget has run out!
    Datarun 1 has ended.
    ```


2. **Explore the results**

    Once the `run` method has finished, we can explore the `results` object which is of type
    `Datarun`.

    **Get a summary of the `Datarun`**:

    ```python
    results.describe()
    ```

    An output similar to this will be printed:

    ```python
    Datarun 1 summary:
        Dataset: 'demos/pollution_1.csv'
        Column Name: 'class'
        Judgment Metric: 'f1'
        Classifiers Tested: 100
        Elapsed Time: 0:00:07.638668
    ```

    **Get a summary of the best classifier**:

    ```python
    results.get_best_classifier()
    ```

    Which will print  the classifiers properties:

    ```python
    Classifier id: 94
    Classifier type: knn
    Params chosen:
        n_neighbors: 13
        leaf_size: 38
        weights: uniform
        algorithm: kd_tree
        metric: manhattan
        _scale: True
    Cross Validation Score: 0.858 +- 0.096
    Test Score: 0.714
    ```

    **Get a dataframe with all the scores**:

    ```python
    scores = results.get_scores()
    ```

    Here we can explore the dataframe as we would like, where the most important field to have
    in mind is the `id` of this classifier in case we would like to recover it.

    If we run `scores.head()` we should get the top 5 classifiers:

    ```python
      cv_judgment_metric cv_judgment_metric_stdev  id test_judgment_metric  rank
    0       0.8584126984             0.0960095737  94         0.7142857143   1.0
    1       0.8222222222             0.0623609564  12         0.6250000000   2.0
    2       0.8147619048             0.1117618135  64         0.8750000000   3.0
    3       0.8139393939             0.0588721670  68         0.6086956522   4.0
    4       0.8067754468             0.0875180564  50         0.6250000000   5.0
    ```


3. **Saving and loading the best classifier**:

    **Saving the best classifier**:
    In order to save the best classifier, the `results` object provides you with a method that
    does it for you:

    ```python
    results.export_best_classifier('path/to/model.pkl')
    ```

    If the classifier has been saved correctly, a message will be printed indicating so:

    ```python
    Classifier 94 saved as path/to/model.pkl
    ```

    If the path that you provide already exists, you can ovewrite it by adding the argument
    `force=True`.

    **Loading the best classifier**:

    Once it's exported you can load it back by calling the `load` method of `Model` that **ATM**
    provides:

    ```python
    from atm import Model

    model = Model.load('path/to/model.pkl')
    ```

    And once you have loaded your model, you can use it's methods to make predictions:

    ```python
    predictions = model.predict(data)
    ```

    **Load the classifier in memory**:

    In case that you want to use the model without exporting it, you can use the `load_model` from
    the `classifier` directly:

    ```python
    classifier = results.get_best_classifier()
    model = classifier.load_model()

    model.predict(data)
    ```

## Customizing ATM's configuration and using your own data

ATM's default configuration is fully controlled by the intern code. Our documentation will
cover the configuration in more detail, but this section provides a brief overview of how
to specify the most important values.


### Setting up a distributed Database

ATM uses a database to store information about datasets, dataruns and classifiers.
It's currently compatible with the SQLite3 and MySQL dialects.

For first-time and casual users, the SQLite3 is used by default without any required
step from the user.

However, if you're planning on running large, distributed, or performance-intensive jobs,
you might prefer using MySQL.

If you do not have a MySQL database already prepared, you can follow the next steps in order
install it and parepare it for ATM:


1. **Install mysql-server**

```bash
sudo apt-get install mysql-server
```

In the latest versions of MySQL no input for the user is required for this step, but
in older versions the installation process will require the user to input a password
for the MySQL root user.

If this happens, keep track of the password that you set, as you will need it in the
next step.

2. **Log into your MySQL instance as root**

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

3. **Create a new Database for ATM**

Once you are logged in, execute the following three commands to create a database
called `atm` and a user also called `atm` with write permissions on it:

```bash
$ mysql> CREATE DATABASE atm;
$ mysql> CREATE USER 'atm'@'localhost' IDENTIFIED BY 'set-your-own-password-here';
$ mysql> GRANT ALL PRIVILEGES ON atm.* TO 'atm'@'localhost';
```

4. **Test your settings**

After you have executed the previous three commands and exited the mysql prompt,
you can test your settings by executing the following command and inputing the
password that you used in the previous step when prompted:

```bash
mysql -u atm -p
```

### Running ATM on your own data

If you want to use the system for your own dataset, convert your data to a CSV file similar
to the example shown above. The format is:

 * Each column is a feature (or the label)
 * Each row is a training example
 * The first row is the header row, which contains names for each column of data
 * A single column (the *target* or *label*) is named `class`

Next, you'll need to use `atm enter_data` to create a `dataset` and `datarun` for your task.

The command line will look for values for each configuration variable in the following places,
in order:

1. Command line arguments
2. Configuration files
3. Defaults specified inside the code.

That means there are two ways to pass configuration to the command.

1. **Using command line arguments**

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

2. **Using YAML configuration files**

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


## REST API Server

**ATM** comes with the possibility to start a server process that enables interacting with
the ModelHub Database via a REST API server that runs over [flask](http://flask.pocoo.org/).

For more details about how to start and use this REST API please check the [API.md](API.md) document.


## Citing ATM

If you use ATM, please consider citing the following paper:

Thomas Swearingen, Will Drevo, Bennett Cyphers, Alfredo Cuesta-Infante, Arun Ross, Kalyan Veeramachaneni. [ATM: A distributed, collaborative, scalable system for automated machine learning.](https://cyphe.rs/static/atm.pdf) *IEEE BigData 2017*, 151-162

BibTeX entry:

```bibtex
@inproceedings{DBLP:conf/bigdataconf/SwearingenDCCRV17,
  author    = {Thomas Swearingen and
               Will Drevo and
               Bennett Cyphers and
               Alfredo Cuesta{-}Infante and
               Arun Ross and
               Kalyan Veeramachaneni},
  title     = {{ATM:} {A} distributed, collaborative, scalable system for automated
               machine learning},
  booktitle = {2017 {IEEE} International Conference on Big Data, BigData 2017, Boston,
               MA, USA, December 11-14, 2017},
  pages     = {151--162},
  year      = {2017},
  crossref  = {DBLP:conf/bigdataconf/2017},
  url       = {https://doi.org/10.1109/BigData.2017.8257923},
  doi       = {10.1109/BigData.2017.8257923},
  timestamp = {Tue, 23 Jan 2018 12:40:42 +0100},
  biburl    = {https://dblp.org/rec/bib/conf/bigdataconf/SwearingenDCCRV17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Related Projects

### BTB

[BTB](https://github.com/hdi-project/btb), for Bayesian Tuning and Bandits, is the core AutoML
library in development under the HDI project. BTB exposes several methods for hyperparameter
selection and tuning through a common API. It allows domain experts to extend existing methods
and add new ones easily. BTB is a central part of ATM, and the two projects were developed in
tandem, but it is designed to be implementation-agnostic and should be useful for a wide range
of hyperparameter selection tasks.

### Featuretools

[Featuretools](https://github.com/featuretools/featuretools) is a python library for automated
feature engineering. It can be used to prepare raw transactional and relational datasets for ATM.
It is created and maintained by [Feature Labs](https://www.featurelabs.com) and is also a part
of the [Human Data Interaction Project](https://hdi-dai.lids.mit.edu/).

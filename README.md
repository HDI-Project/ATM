<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“ATM” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>



[![CircleCI][circleci-img]][circleci-url]
[![Travis](https://travis-ci.org/HDI-Project/ATM.svg?branch=master)](https://travis-ci.org/HDI-Project/ATM)
[![Coverage status][codecov-img]][codecov-url]
[![Documentation][rtd-img]][rtd-url]

[circleci-img]: https://circleci.com/gh/HDI-Project/ATM.svg?style=shield
[circleci-url]: https://circleci.com/gh/HDI-Project/ATM
[travis-img]: https://travis-ci.org/HDI-Project/ATM.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/ATM
[pypi-img]: https://img.shields.io/pypi/v/atm.svg
[pypi-url]: https://pypi.python.org/pypi/atm
[codecov-img]: https://codecov.io/gh/HDI-project/ATM/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/HDI-project/ATM
[rtd-img]: https://readthedocs.org/projects/atm/badge/?version=latest
[rtd-url]: http://atm.readthedocs.io/en/latest/


# ATM - Auto Tune Models

- Free software: MIT license
- Documentation: http://atm.readthedocs.io/en/latest/


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

## Setup/Installation

This section describes the quickest way to get started with ATM on a machine running Ubuntu Linux.
We hope to have more in-depth guides in the future, but for now, you should be able to substitute
commands for the package manager of your choice to get ATM up and running on most modern
Unix-based systems.

ATM is compatible with and has been tested on Python 2.7, 3.5, and 3.6.


1. **Install a database**

   You will need to install the libmysqlclient-dev package (for sqlalchemy)

   ```
   sudo apt install libmysqlclient-dev
   ```

   and at least one of the following databases.

   - for SQLite (simpler):

   ```
   sudo apt install sqlite3
   ```

   - for MySQL:

   ```
   sudo apt install mysql-server mysql-client
   ```

2. **Install ATM**

    To get started with **ATM**, we recommend using `pip`:

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
We will use a featurized dataset, already saved in `data/test/pollution_1.csv`.
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

## Customizing ATM's configuration and using your own data

ATM's default configuration is fully controlled by the intern code. Our documentation will
cover the configuration in more detail, but this section provides a brief overview of how
to specify the most important values.

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

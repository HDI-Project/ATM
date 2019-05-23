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

- License: MIT
- Documentation: https://hdi-project.github.io/ATM/
- Homepage: https://github.com/HDI-Project/ATM

## Overview

**ATM** is a distributed, scalable AutoML system designed with ease of use in mind.

For a given classification problem, ATM's goal is to find

1. a classification *method*, such as *decision tree*, *support vector machine*,
or *random forest*, and
2. a set of *hyperparameters* for that method

which generate the best classifier possible.

## Getting Started

### Requirements

#### Python

**ATM** has been developed and been tested on [Python 2.7, 3.5, and 3.6](https://www.python.org)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **ATM** is run.

It's recommended that you create a virtualenv. In this example, we will create it using
[VirtualEnvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/):

```bash
mkvirtualenv -p $(which python3.6) -a $(pwd) ATM
```

### Install

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **ATM**:

```bash
pip install atm
```

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install`:

```bash
git clone git@github.com:HDI-Project/ATM.git
cd ATM
make install
```

For development, use the following command instead, which will install some additional dependencies
for code linting and testing.

```bash
make install-develop
```


## Quickstart

In this short tutorial we will guide you through a series of steps that will help you getting
started with **ATM**.

### Training data

At the moment, **ATM**, works with classification problems, and all it requieres as a training data
is a dataset with pre-extracted feature vectors and labels as a CSV file.

All you will need to provide to **ATM** in order to generate models, is the path to such a CSV file.

### Using ATM with Python

#### 1. Generate the demo data

In order to get started, you will need some demo data as described before. ATM provides you with
a simple command that will download a few datasets in order to run this Quickstart guide.

```python
from atm import data

demo_datasets = data.get_demos()
```

Inside the `demo_datasets` variable, we will store the returned paths to this datasets:

```python
demo_datasets

{
    'iris.data.csv': 'demos/iris.data.csv',
    'pollution_1.csv': 'demos/pollution_1.csv',
    'pitchfork_genres.csv': 'demos/pitchfork_genres.csv'
}
```

#### 2. Auto Tune Model over a CSV file

In order to Auto Tune Models over a CSV file, first you will need to create an instance of `ATM`:

```python
from atm import ATM

atm = ATM()
```

This will create an instance with the default settings for `ATM`.

Once you have the instance ready, you can use the method `run` by giving the argument `train_path`
to where the `CSV` file that contains your dataset (in this case we will use `pollution_1`) is
located:

```python
path_to_csv = demo_datasets.get('pollution_1.csv')  # this is equal to 'demos/pollution_1.csv'

results = atm.run(train_path=path_to_csv)
```

This process will display a progress bar during it's execution to keep track on the progress that
**ATM** is doing.

```python
Processing dataset demos/pollution_1.csv
100%|##########################| 100/100 [00:10<00:00,  6.09it/s]
Classifier budget has run out!
Datarun 1 has ended.
```

Once this process has ended, a message will print that the `Datarun` has ended. Then we can
explore the `results` object.

#### 2. Explore the results

Once the `run` process has finished, we can explore the `results` object with the methods that it
offers.

- Get a summary of the `Datarun`:

    ```python
    results.describe()
    ```

    This will print a short description of this `datarun`:

    ```python
    Datarun 1 summary:
        Dataset: 'demos/pollution_1.csv'
        Column Name: 'class'
        Judgment Metric: 'f1'
        Classifiers Tested: 100
        Elapsed Time: 0:00:07.638668
    ```

- Get a summary of the best classifier:

    ```python
    results.get_best_classifier()
    ```

    This will print the best classifier that was found during this run:

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

- Get a dataframe with all the scores:

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

#### 3. Saving and loading the best classifier:

##### Saving the best classifier:

In order to save the best classifier, the `results` object, provides you with
`export_best_classifier` method, that takes as an argument the path where you would like to save
this classifier:

```python
results.export_best_classifier('path/to/model.pkl')
```

If the classifier has been saved correctly, a message will be printed indicating so:

```python
Classifier 94 saved as path/to/model.pkl
```

If the path that you provide already exists, you can ovewrite it by adding the argument
`force=True`.

##### Loading a saved classifier:

Once it's exported you can load it back by calling the `load` method of `Model` that **ATM**
provides, and takes as an argument the path to where the model has been saved:

```python
from atm import Model

model = Model.load('path/to/model.pkl')
```

And once you have loaded your model, you can use it's methods to make predictions:

```python
import pandas as pd

data = pd.read_csv(demo_datasets.get('pollution_1.csv'))

predictions = model.predict(data[0])
```

**Note** data is the dataframe that we used to train our model, at this step you should be using
data that you would like to predict.

##### Load the classifier in memory:

In case that you want to use the model without exporting it, you can use the `load_model` from
the `classifier` directly:

```python
classifier = results.get_best_classifier()

model = classifier.load_model()

predictions = model.predict(data[0])
```

#### Using your own data with the `run` method

If you would like to use your datasets with **ATM**, please reffer to
[CUSTOM USAGE](CUSTOM_USAGE.md) which contains more information about the data format accepted,
`run` method arguments and custom configuration.

### Command Line

**ATM** provides a simple command line utility that will allow you to run a process of AutoML over
a `CSV` file which follows the format specified inside [CUSTOM_USAGE.md](CUSTOM_USAGE.md).

In this example, we will use the default values that are provided in the code.

#### 1. Generate the demo data

**ATM** command line allows you to generate the demo data that we will be using through this steps
by running the following command:

```bash
atm get_demos
```

A print on your console with the generated demo datasets will appear:

```bash
Generating file demos/iris.data.csv
Generating file demos/pollution_1.csv
Generating file demos/pitchfork_genres.csv
```

#### 2. Create a dataset and generate it's dataruns

Once you have generated the demo datasets, now it's time to create a `dataset` object inside the
database. Our command line also triggers the generation of `datarun` objects for this dataset in
order to automate this process as much as possible:

```bash
atm enter_data
```

If you run this command, you will create a dataset with the default values, which is using the
`pollution_1.csv` dataset.

A print, with similar information to this, should be printed:

```bash
method logreg has 6 hyperpartitions
method dt has 2 hyperpartitions
method knn has 24 hyperpartitions
Dataruns created. Summary:
	Dataset ID: 1
	Training data: demos/pollution_1.csv
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

#### 3. Start a worker

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

## REST API Server

**ATM** comes with the possibility to start a server process that enables interacting with
the ModelHub Database via a REST API server that runs over [flask](http://flask.pocoo.org/).

For more details about how to start and use this REST API please check the [API.md](API.md) document.

## Credits

### Contributors

* Bennett Cyphers <bcyphers@mit.edu>
* Thomas Swearingen <swearin3@msu.edu>
* Kalyan Veeramachaneni <kalyan@mit.edu>
* Laura Gustafson <lgustaf@mit.edu>
* Carles Sala <csala@csail.mit.edu>
* Plamen Valentinov <plamen@pythiac.com>
* Micah Smith <micahjsmith@gmail.com>
* Kiran Karra <kiran.karra@gmail.com>
* Max Kanter <kmax12@gmail.com>
* Alfredo Cuesta-Infante <alfredo.cuesta@urjc.es>
* Favio André Vázquez <favio.vazquezp@gmail.com>
* Matteo Hoch <minime@hochweb.com>


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

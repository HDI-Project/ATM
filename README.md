<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“ATM” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>



[![CircleCI](https://circleci.com/gh/HDI-Project/ATM.svg?style=shield)](https://circleci.com/gh/HDI-Project/ATM)
[![Travis](https://travis-ci.org/HDI-Project/ATM.svg?branch=master)](https://travis-ci.org/HDI-Project/ATM)
[![PyPi Shield](https://img.shields.io/pypi/v/atm.svg)](https://pypi.python.org/pypi/atm)
[![Coverage Status](https://codecov.io/gh/HDI-project/ATM/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-project/ATM)
[![Downloads](https://pepy.tech/badge/atm)](https://pepy.tech/project/atm)


# ATM - Auto Tune Models

- License: MIT
- Documentation: https://HDI-Project.github.io/ATM/
- Homepage: https://github.com/HDI-Project/ATM

# Overview

Auto Tune Models (ATM) is an AutoML system designed with ease of use in mind. In short, you give
ATM a classification problem and a dataset as a CSV file, and ATM will try to build the best model
it can. ATM is based on a [paper](https://dai.lids.mit.edu/wp-content/uploads/2018/02/atm_IEEE_BIgData-9-1.pdf)
of the same name, and the project is part of the [Human-Data Interaction (HDI) Project](https://hdi-dai.lids.mit.edu/) at MIT.


# Install

## Requirements

**ATM** has been developed and tested on [Python 2.7, 3.5, and 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **ATM** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **ATM**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) atm-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source atm-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **ATM**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **ATM**:

```bash
pip install atm
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:HDI-Project/ATM.git
cd ATM
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

First, please head to [the GitHub page of the project](https://github.com/HDI-Project/ATM)
and make a fork of the project under you own username by clicking on the **fork** button on the
upper right corner of the page.

Afterwards, clone your fork and create a branch from master with a descriptive name that includes
the number of the issue that you are going to work on:

```bash
git clone git@github.com:{your username}/ATM.git
cd ATM
git branch issue-xx-cool-new-feature master
git checkout issue-xx-cool-new-feature
```

Finally, install the project with the following command, which will install some additional
dependencies for code linting and testing.

```bash
make install-develop
```

Make sure to use them regularly while developing by running the commands `make lint` and `make test`.


# Data Format

ATM input is always a CSV file with the following characteristics:

* It uses a single comma, `,`, as the separator.
* Its first row is a header that contains the names of the columns.
* There is a column that contains the target variable that will need to be predicted.
* The rest of the columns are all variables or features that will be used to predict the target column.
* Each row corresponds to a single, complete, training sample.

Here are the first 5 rows of a valid CSV with 4 features and one target column called `class` as an example:

```
feature_01,feature_02,feature_03,feature_04,class
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
```

This CSV can be passed to ATM as local filesystem path but also as a complete AWS S3 Bucket and
path specification or as a URL.

You can find a collection of demo datasets in the [atm-data S3 Bucket in AWS](https://atm-data.s3.amazonaws.com/index.html).


# Quickstart

In this short tutorial we will guide you through a series of steps that will help you getting
started with **ATM** by exploring its Python API.

## 1. Get the demo data

The first step in order to run **ATM** is to obtain the demo datasets that will be used in during
the rest of the tutorial.

For this demo we will be using the pollution csv from the atm-data bucket, which you can download with your browser
[from here](https://atm-data.s3.amazonaws.com/pollution_1.csv), or using the following command:

```bash
wget https://atm-data.s3.amazonaws.com/pollution_1.csv
```

## 2. Create an ATM instance

The first thing to do after obtaining the demo dataset is creating an ATM instance.

```python
from atm import ATM

atm = ATM()
```

By default, if the ATM instance is without any arguments, it will create an SQLite database
called `atm.db` in your current working directory.

If you want to connect to a SQL database instead, or change the location of your SQLite database,
please check the [API Reference](https://hdi-project.github.io/ATM/api/atm.core.html)
for the complete list of available options.

## 3. Search for the best model

Once you have the **ATM** instance ready, you can use the method `atm.run` to start
searching for the model that better predicts the target column of your CSV file.

This function has to be given the path to your CSV file, which can be a local filesystem path, an URL to
and HTTP or S3 resource.

For example, if we have previously downloaded the [pollution_1.csv](https://atm-data.s3.amazonaws.com/pollution_1.csv)
file inside our current working directory, we can call `run` like this:

```python
results = atm.run(train_path='pollution_1.csv')
```

Alternatively, we can use the HTTPS URL of the file to have ATM download the CSV for us:

```python
results = atm.run(train_path='https://atm-data.s3.amazonaws.com/pollution_1.csv')
```

As the last option, if we have the file inside an S3 Bucket, we can download it by passing an URI
in the `s3://{bucket}/{key}` format:

```python
results = atm.run(train_path='s3://atm-data/pollution_1.csv')
```

In order to make this work with a Private S3 Bucket, please make sure to having configured your
[AWS credentials file](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html),
or to having created your `ATM` instance passing it the `access_key` and `secret_key` arguments.

This `run` call will start what is called a `Datarun`, and a progress bar will be displayed
while the different models are tested and tuned.

```python
Processing dataset demos/pollution_1.csv
100%|##########################| 100/100 [00:10<00:00,  6.09it/s]
```

Once this process has ended, a message will print that the `Datarun` has ended. Then we can
explore the `results` object.

## 4. Explore the results

Once the Datarun has finished, we can explore the `results` object in several ways:

**a. Get a summary of the Datarun**

The `describe` method will return us a summary of the Datarun execution:

```python
results.describe()
```

This will print a short description of this Datarun similar to this:

```python
Datarun 1 summary:
    Dataset: 'demos/pollution_1.csv'
    Column Name: 'class'
    Judgment Metric: 'f1'
    Classifiers Tested: 100
    Elapsed Time: 0:00:07.638668
```

**b. Get a summary of the best classifier**

The `get_best_classifier` method will print information about the best classifier that was found
during this Datarun, including the method used and the best hyperparameters found:

```python
results.get_best_classifier()
```

The output will be similar to this:

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

**c. Explore the scores**

The `get_scores` method will return a `pandas.DataFrame` with information about all the
classifiers tested during the Datarun, including their cross validation scores and
the location of their pickled models.

```python
scores = results.get_scores()
```

The contents of the scores dataframe should be similar to these:

```python
  cv_judgment_metric cv_judgment_metric_stdev  id test_judgment_metric  rank
0       0.8584126984             0.0960095737  94         0.7142857143   1.0
1       0.8222222222             0.0623609564  12         0.6250000000   2.0
2       0.8147619048             0.1117618135  64         0.8750000000   3.0
3       0.8139393939             0.0588721670  68         0.6086956522   4.0
4       0.8067754468             0.0875180564  50         0.6250000000   5.0
...
```

## 5. Make predictions

Once we have found and explored the best classifier, we will want to make predictions with it.

In order to do this, we need to follow several steps:

**a. Export the best classifier**

The `export_best_classifier` method can be used to serialize and save the best classifier model
using pickle in the desired location:

```python
results.export_best_classifier('path/to/model.pkl')
```

If the classifier has been saved correctly, a message will be printed indicating so:

```python
Classifier 94 saved as path/to/model.pkl
```

If the path that you provide already exists, you can ovewrite it by adding the argument
`force=True`.

**b. Load the exported model**

Once it is exported you can load it back by calling the `load` method from the `atm.Model`
class and passing it the path where the model has been saved:

```python
from atm import Model

model = Model.load('path/to/model.pkl')
```

Once you have loaded your model, you can pass new data to its `predict` method to make
predictions:

```python
import pandas as pd

data = pd.read_csv(demo_datasets['pollution'])

predictions = model.predict(data.head())
```


# What's next?

For more details about **ATM** and all its possibilities and features, please check the
[documentation site](https://HDI-Project.github.io/ATM/).

There you can learn more about its [Command Line Interface](https://hdi-project.github.io/ATM/cli.html)
and its [REST API](https://hdi-project.github.io/ATM/rest.html), as well as
[how to contribute to ATM](https://HDI-Project.github.io/ATM/community/contributing.html)
in order to help us developing new features or cool ideas.

# Credits

ATM is an open source project from the Data to AI Lab at MIT which has been built and maintained
over the years by the following team:

* Bennett Cyphers <bcyphers@mit.edu>
* Thomas Swearingen <swearin3@msu.edu>
* Carles Sala <csala@csail.mit.edu>
* Plamen Valentinov <plamen@pythiac.com>
* Kalyan Veeramachaneni <kalyan@mit.edu>
* Micah Smith <micahjsmith@gmail.com>
* Laura Gustafson <lgustaf@mit.edu>
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

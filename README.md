delphi
====

## Setup

Only Unix machines have been tested with Delphi. 

### 1) Install `virtualenvwrapper`

```bash
$ sudo apt-get install python-pip
$ sudo pip install virtualenv 
$ sudo pip install virtualenvwrapper
```

Add into ~/.bash_profil:e

```bash
# for vitrualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper_lazy.sh
```

Create the virtual environment and:

```bash
$ mkvirtualenv delphi-env
$ workon delphi-env
(delphi-env)$ 
```

## 2) Install MySQL 

For ubuntu, this is easy. For Mac OSX, not so much. 

Ubuntu:

```bash
$ sudo apt-get install git python-dev mysql-server mysql-client gfortran libatlas-base-dev libmysqlclient-dev build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base libfreetype6-dev libxft-dev
```

and follow instructions. To install on Mac OSX, go [here](http://dev.mysql.com/downloads/mysql/5.5.html#macosx-dmg).

## 3) Install Python libraries

```bash
$ workon delphi-env
(delphi-env)$ pip install -r config/requirements.txt
```

and verify the libraries are installed:

```bash
$ python
>>> import numpy as np
>>> import scipy
>>> import sklearn
>>> import pandas as pd
```

## 4) Create a MySQL database and load the tables

```bash
$ mysql -u <username> -p <password-if-needed>
mysql> create database delphi_db;
mysql> exit;
$ mysql -u <username> -p <password-if-needed> delphi_db < config/hyperdelphi.sql
```

## Compiling Datasets for Classification

Folder structure for `delphi/data`:

	data
		binary (binary classification datasets)
		multiclass (mutliclass)
		processed (Delphi generated ready-to-model CSVs. Don't manually mess with things here, generally)
		
ALFA has now begun to compile datasets (supervised classification for now) in a standardized (CSV) format such that different teams can store their data in a single place.

Each CSV file needs to:

* Have the first line of the file be headers with strings as the feature names, and the class column named "class". If the features aren't named (ie, image or SVD or PCA data), then anything will do (but see below for a small script to generate nice feature names).
* Should have N + 1 rows (1 header + N examples)
* Should have D + 1 features (1 class label + D features per example)

Here's a handy Python script to create a CSV header line for data that doesn't have feaure names:

```python
def CreateCSVHeader(n_features, name, class_label_name):
    """
        Creates a CSV header like:
            "<class_label_name>, <name>1, <name>2, ..., <name><n_features>"

        Example:
            print CreateCSVHeader(64, "pixel", "class")
    """
    separator = ","
    header_row_string = separator.join(
        [x + str(y) for (x, y) in
            zip([name for i in range(n_features)], range(1, n_features + 1, 1))])
    return separator.join([class_label_name, header_row_string])
```

## Running Data! (locally)

Now you'd like to start a run. First off, you'll need to add your `datarun` to the database. A datarun consists of all the parameters for a single experiment run, including where the find the data, what the budget is for number of learners to train, the majoirty class benchmark, and other things. The datarun ID in the database also ties together the `hyperpartitions` (frozen sets) which delinate how Delphi can explore different subtypes of classifiers to maximize their performance. 

Tweaking parameters of your run is quite normal. Here I've copied the header of the `run.py` file for you to get an idea of the parameters here. 

```python
"""
Add here the algorithm codes you'd like to run and compare. Look these up in the 
`algorithms` table in the MySQL database, or alternatively, in the config/hyperdelphi.sql 
file in this repository. You must spell them correctly!

Add each algorithm as a string to the list. 

Notes:
	- SVMs (classify_svm) can take a long time to train. It's not an error. It's just part of what
	happens when the algorithm happens to explore a crappy set of parameters on a powerful algo like this. 
	- SGDs (classify_sgd) can sometimes fail on certain parameter settings as well. Don't worry, they 
	train SUPER fast, and the worker.py will simply log the error and continue. 
"""
algorithm_codes = ['classify_sgd', 'classify_dt', 'classify_dbn']


"""
This is where you list CSV files to train on. Follow the CSV conventions on the ALFA online
wiki page. 

Note that if you want Delphi to randomly create a train/test split, just add a single string 
CSV path to the list. If you've already got it split into two files, name them
	
	DATANAME_train.csv
	DATANAME_test.csv 

and add them as a tuple of string, with the TRAINING set as the first item and the TESTING 
as the second string in the tuple. 
"""
csvfiles = [
	"data/binary/banknote.csv",
	# ("data/binary/congress_train.csv", "data/binary/congress_test.csv"),
]


"""
How many learners would you like Delphi to train in this run? Be aware some classifiers are very
quick to train and others take a long time depending on the size and dimensionality of the data. 
"""
nlearners = 500


"""
Should there be a 
	"learner", or
	"walltime"

budget? You decide here. 
"""
budget_type = "learner"


"""
How should Delphi sample a frozen set that it must explore?
	- uniform: pick randomly! (baseline)
	- gp_ei: Gaussian Process expected improvement criterion
	- gp_eitime: Gaussian Process expected improvement criterion 
				per unit time

The number in the second part of the tuple is the `r_min` parameter. Consult
the thesis to understand what those mean, but essentially: 

	if (num_learners_trained_in_hyperpartition >= r_min)
		# train using sample criteria 
	else
		# train using uniform (baseline)
"""
sample_selectors = [
	# sample selection, r_min
	#("uniform", -1),
	("gp_ei", 3),
	#("gp_eitime", 3),
]


"""
How should Delphi select a hyperpartition (frozen set) from the current options it has? 

Again, each is a different method, consult the thesis. The second numerical entry in the
tuple is similar to r_min, except it is called k_window and determines how much "history"
Delphi considers for certain frozen selection logics. 
"""
frozen_selectors = [
	# frozen selection, k_window
	#("uniform", -1),
	#("ucb1", -1),
	#("bestkvel", 5),
	("purebestkvel", 5),
	#("hieralg", -1),
]
	

"""
What is the priority of this run? Higher is more important. 
"""
priority = 10


"""
What metric should Delphi use to score? Keep this as "cv".
"""
metric = "cv"
```

Simply edit `run.py` before you start your run. 

To start, run `run.py`:

```bash
$ python run.py
```

You should see a great deal of description about what was loaded, the frozen hyperpartitions created, etc. You might also see something demanding you upload your data in CSV files to an HTTP address, but don't worry about that for now. That only matters when deploying onto a cloud. 

The next step is to run a (or multiple) workers to actually train models and report progress back to the MySQL DataHub. 

```bash
$ export DELPHI_ENV=LOCAL
$ python worker.py
```

This will create a worker that will automatically work on any uncompleted frozen set / datarun in the database. You might notice that in `run.py` you can change not only the type of budget (number of learners vs. walltime), but also the priority. The HIGHER the priority means that any workers in the pool will try to finish the higher prioritied dataruns FIRST. This is helpful if your needs change (maybe a paper deadline!) and you want to switch the focus of your worker cloud to a different experiment. 

## Running Data! (on the cloud!)

Navigate to the [Nimbus portal for creating nodes](https://nimbus.csail.mit.edu/horizon/project/instances/). Here, create a few nodes with the following specifications:

* Boot from snapshot `scikit-postgres-user-setup-ALL`
* Choose `m1.4core` for the size
* Select a keypair you have access to and is placed in your `~/.ssh/` folder with `0644` permissions
* Name the instance `delphi-worker` (important)

Once you've launched it, wait until the status is "Active". If you like, you can SSH into the node:

```bash
ssh ubuntu@<IP address from Nimbus> -i <path-to-pem-file>
```

Uses `fabric` to deploy nodes and apply updates.

	$ fab openstack deploy



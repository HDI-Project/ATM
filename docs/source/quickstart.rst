Quick-start guide
=================

This page is a quick tutorial to help you get ATM up and running for the first
time. We'll use a featurized dataset for a binary classification problem,
already saved in ``atm/data/test/pollution_1.csv``. This is one of the datasets
available on `openml.org <https://openml.org>`_.  More information about the
data can be found `here <https://www.openml.org/d/542>`_. 

Our goal is predict mortality using the metrics associated with the air
pollution. Below we show a snapshot of the csv file. The dataset has 15
features, all numeric, and and a binary label column called "class".

.. code-block:: none

	PREC 	JANT 	JULT 	OVR65 	POPN 	EDUC 	HOUS 	DENS 	NONW 	WWDRK 	POOR 	HC 	NOX 	SO@ 	HUMID 	class
	35 	23 	72 	11.1 	3.14 	11 	78.8 	4281 	3.5 	50.7 	14.4 	8 	10 	39 	57 	1
	44 	29 	74 	10.4 	3.21 	9.8 	81.6 	4260 	0.8 	39.4 	12.4 	6 	6 	33 	54 	1
	47 	45 	79 	6.5 	3.41 	11.1 	77.5 	3125 	27.1 	50.2 	20.6 	18 	8 	24 	56 	1
	43 	35 	77 	7.6 	3.44 	9.6 	84.6 	6441 	24.4 	43.7 	14.3 	43 	38 	206 	55 	1
	53 	45 	80 	7.7 	3.45 	10.2 	66.8 	3325 	38.5 	43.1 	25.5 	30 	32 	72 	54 	1
	43 	30 	74 	10.9 	3.23 	12.1 	83.9 	4679 	3.5 	49.2 	11.3 	21 	32 	62 	56 	0
	45 	30 	73 	9.3 	3.29 	10.6 	86 	2140 	5.3 	40.4 	10.5 	6 	4 	4 	56 	0
	.. 	.. 	.. 	... 	.... 	.... 	... 	.... 	.. 	.... 	.... 	.. 	.. 	.. 	.. 	.
	.. 	.. 	.. 	... 	.... 	.... 	... 	.... 	.. 	.... 	.... 	.. 	.. 	.. 	.. 	.
	.. 	.. 	.. 	... 	.... 	.... 	... 	.... 	.. 	.... 	.... 	.. 	.. 	.. 	.. 	.
	37 	31 	75 	8 	3.26 	11.9 	78.4 	4259 	13.1 	49.6 	13.9 	23 	9 	15 	58 	1
	35 	46 	85 	7.1 	3.22 	11.8 	79.9 	1441 	14.8 	51.2 	16.1 	1 	1 	1 	54 	0

Create a datarun
----------------

Before we can train any classifiers, we need to create a datarun. In ATM, a
datarun is a single logical machine learning task. The ``enter_data.py`` script
will set up everything you need.::

$ python scripts/enter_data.py

The first time you run it, the above command will create a ModelHub database, a
dataset, and a datarun. If you run it without any arguments, it will load
configuration from the default values defined in ``atm/config.py``. By default,
it will create a new SQLite3 database at ./atm.db, create a new dataset instance
which refers to the data at ``atm/data/test/pollution_1.csv``, and create a
datarun instance which points to that dataset. 

The command should produce output that looks something like this:::

	method logreg has 6 hyperpartitions
	method dt has 2 hyperpartitions
	method knn has 24 hyperpartitions
	Data entry complete. Summary:
					Dataset ID: 1
					Training data: /home/bcyphers/work/fl/atm/atm/data/test/pollution_1.csv
					Test data: None
					Datarun ID: 1
					Hyperpartition selection strategy: uniform
					Parameter tuning strategy: uniform
					Budget: 100 (classifier)

The datarun you just created will train classifiers using the "logreg"
(logistic regression), "dt" (decision tree), and "knn" (k nearest neighbors)
methods. It is using the "uniform" strategy for both hyperpartition selection
and parameter tuning, meaning it will choose parameters uniformly at random. It
has a budget of 100 classifiers, meaning it will train and test 100 models
before completing. More info about what is stored in the database, and
what the fields of the datarun control, can be found `here <database.html>`_.

The most important piece of information is the datarun ID. You'll need to
reference that when you want to actually compute on the datarun.

Execute the datarun
-------------------

An ATM *worker* is a process that connects to a ModelHub, asks it what dataruns
need to be worked on, and trains and tests classifiers until all the work is
done. To run one, use the following command::

$ python scripts/worker.py 

This will start a process that builds classifiers, tests them, and saves them to
the ./models/ directory. As it runs, it should print output indicating which
hyperparameters are being tested, the performance of each classifier it builds,
and the best overall performance so far. One round of training looks like this::

  Computing on datarun 1
  Selector: <class 'btb.selection.uniform.Uniform'>
  Tuner: <class 'btb.tuning.uniform.Uniform'>
  Chose parameters for method "knn":
          _scale = True
          algorithm = brute
          metric = euclidean
          n_neighbors = 8
          weights = distance
          Judgment metric (f1, cv): 0.813 +- 0.081
  New best score! Previous best (classifier 24): 0.807 +- 0.284
  Saving model in: models/pollution_1-62233d75.model
  Saving metrics in: metrics/pollution_1-62233d75.metric
  Saved classifier 63.

And that's it! You're executing your first datarun, traversing the vast space
of hyperparameters to find the absolute best model for your problem. You can
break out of the worker with Ctrl+C and restart it with the same command; it
will pick up right where it left off. You can also run the command
simultaneously in different terminals to parallelize the work -- all workers
will refer to the same ModelHub database. 

Occassionally, a worker will encounter an error in the process of building and
testing a classifier. Don't worry: when this happens, the worker will print
error data to the terminal, log the error in the database, and move on to the
next classifier.

When all 100 classifiers in your budget have been built, the datarun is
finished! All workers will exit gracefully.

::

  Classifier budget has run out!
  Datarun 1 has ended.
  No dataruns found. Exiting.

You can then load the best classifier from the datarun and use it to make
predictions on new datapoints.

>>> from atm.database import Database
>>> db = Database(dialect='sqlite', database='atm.db')
>>> model = db.load_model(classifier_id=110)
>>> import pandas as pd
>>> data = pd.read_csv('atm/data/test/pollution_1.csv')
>>> model.predict(data[0])

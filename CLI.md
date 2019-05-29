# Command Line Interface

**ATM** provides a simple command line client that will allow you to run ATM directly
from your terminal by simply passing it the path to a CSV file.

In this example, we will use the default values that are provided in the code, which will use
the `pollution.csv` that is being generated with the demo datasets by ATM.

## 1. Get the demo data

The first step in order to run **ATM** is to obtain the demo datasets that will be used in during
the rest of the tutorial.

For this demo we will be using the pollution csv from the demos bucket, which you can download from
[here](https://atm-data.s3.amazonaws.com/pollution_1.csv).


## 2. Create a dataset and generate it's dataruns

Once you have obtained your demo dataset, now it's time to create a `dataset` object inside the
database. Our command line also triggers the generation of `datarun` objects for this dataset in
order to automate this process as much as possible:

```bash
atm enter_data --train-path path/to/pollution_1.csv
```

Have in mind that `--train-path` argument can be a local path, an URL link to the CSV file or an
S3 bucket like link.

If you run this command, you will create a dataset with the default values, which is using the
`pollution_1.csv` dataset from the demo datasets.

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

## 3. Start a worker

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

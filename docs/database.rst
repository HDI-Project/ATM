Guide to the ModelHub database
==============================

The ModelHub database is what ATM uses to save state about ongoing jobs,
datasets, and previously-generated models. It allows multiple workers on
multiple machines to collaborate on a single task, regardless of failures or
interruptions. The ideas behind ModelHub are described in the corresponding
`paper <https://cyphe.rs/static/atm.pdf>`_, although the structure described
there does not match up one-to-one with the ModelHub implemented in
``atm/database.py``. This page gives a brief overview of the structure of the
ModelHub database as implemented and how it compares to the version in the
paper.

Datasets
--------
A Dataset represents a single set of data which can be used to train and test
models by ATM. The table stores information about the location of the data as
well as metadata to help with analysis.

- ``dataset_id`` (Int): Unique identifier for the dataset.
- ``name`` (String): Identifier string for a classification technique.
- ``description`` (String): Human-readable description of the dataset.
    - not described in the paper
- ``train_path`` (String): Location of the dataset train file.
- ``test_path`` (String): Location of the dataset test file.
- ``class_column`` (String): Name of the class label column.

The metadata fields below are not described in the paper.

- ``n_examples`` (Int): Number of samples (rows) in the dataset.
- ``k_classes`` (Int): Number of classes in the dataset.
- ``d_features`` (Int): Number of features in the dataset.
- ``majority`` (Number): Ratio of the number of samples in the largest class to
  the number of samples in all other classes.
- ``size_kb`` (Int): Approximate size of the dataset in KB.


Dataruns
--------
A Datarun is a single logical job for ATM to complete. The Dataruns table
contains a reference to a dataset, configuration for ATM and BTB, and
state information.

- ``datarun_id`` (Int): Unique identifier for the datarun.
- ``dataset_id`` (Int): ID of the dataset associated with this datarun.
- ``description`` (String): Human-readable description of the datarun.
    - not in the paper

BTB configuration:

- ``selector`` (String): Selection technique for hyperpartitions.
    - called "hyperpartition_selection_scheme" in the paper
- ``k_window`` (Int): The number of previous classifiers the selector will
  consider, for selection techniques that set a limit of the number of
  historical runs to use.
    - called "t\ :sub:`s`" in the paper
- ``tuner`` (String): The technique that BTB will use to choose new continuous
  hyperparameters.
    - called "hyperparameters_tuning_scheme" in the paper
- ``r_minimum`` (Int): The number of random runs that must be performed in each
  hyperpartition before allowing Bayesian optimization to select parameters.
- ``gridding`` (Int): If this value is set to a positive integer, each
  numeric hyperparameter will be chosen from a set of ``gridding`` discrete,
  evenly-spaced values. If set to 0 or NULL, values will be chosen from the
  full, continuous space of possibilities.
    - not in the paper

ATM configuration:

- ``priority`` (Int): Run priority for the datarun. If multiple unfinished
  dataruns are in the ModelHub at once, workers will process higher-priority
  runs first.
- ``budget_type`` (Enum): One of ["learner", "walltime"]. If this is "learner",
  only ``budget`` classifiers will be trained; if "walltime", classifiers will
  only be trained for ``budget`` minutes total.
- ``budget`` (Int): The maximum number of classifiers to build, or the maximum
  amount of time to train classifiers (in minutes).
    - called "budget_amount" in the paper
- ``deadline`` (DateTime): If provided, and if ``budget_type`` is set to
  "walltime", the datarun will run until this absolute time. This overrides the
  ``budget`` column.
    - not in the paper
- ``metric`` (String): The metric by which to score each classifier for
  comparison purposes. Can be one of ["accuracy", "cohen_kappa", "f1",
  "roc_auc", "ap", "mcc"] for binary problems, or ["accuracy", "rank_accuracy",
  "cohen_kappa", "f1_micro", "f1_macro", "roc_auc_micro", "roc_auc_macro"] for
  multiclass problems
    - not in the paper
- ``score_target`` (Enum): One of ["cv", "test", "mu_sigma"]. Determines how the
  final comparative metric (the *judgment metric*) is calculated.
    - "cv" (cross-validation): the judgment metric is the average of a 5-fold
      cross-validation test.
    - "test": the judgment metric is computed on the test data.
    - "mu_sigma": the judgment metric is the lower error bound on the mean CV
      score.
  - not in the paper

State information:

- ``start_time`` (DateTime): Time the DataRun began.
- ``end_time`` (DateTime): Time the DataRun was completed.
- ``status`` (Enum): Indicates whether the run is pending, in progress, or has
  been finished. One of ["pending", "running", "complete"].
    - not in the paper


Hyperpartitions
---------------
A Hyperpartition is a fixed set of categorical hyperparameters which defines a
space of numeric hyperparameters that can be explored by a tuner. ATM uses BTB
selectors to choose among hyperpartitions during a run. Each hyperpartition
instance must be associated with a single datarun; the performance of a
hyperpartition in a previous datarun is assumed to have no bearing on its
performance in the future.

- ``hyperparition_id`` (Int): Unique identifier for the hyperparition.
- ``datarun_id`` (Int): ID of the datarun associated with this hyperpartition.
- ``method`` (String): Code for, or path to a JSON file describing, this
  hyperpartition's classification method (e.g. "svm", "knn").
- ``categoricals`` (Base64-encoded object): List of categorical hyperparameters
  whose values are fixed to define this hyperpartition.
    - called "partition_hyperparameter_values" in the paper
- ``tunables`` (Base64-encoded object): List of continuous hyperparameters which
  are free; their values must be selected by a Tuner.
    - called "conditional_hyperparameters" in the paper
- ``constants`` (Base64-encoded object): List of categorical or continuous
  parameters whose values are always fixed. These do not define the
  hyperpartition, but their values must be passed to the classification method
  to fully parameterize it.
    - not in the paper
- ``status`` (Enum): Indicates whether the hyperpartition has caused too many
  classifiers to error, or whether the grid for this partition has been fully
  explored. One of ["incomplete", "gridding_done", "errored"].
    - not in the paper


Classifiers
-----------
A Classifier represents a single train/test run using a method and a set of hyperparameters with a particular dataset.

- ``classifier_id`` (Int): Unique identifier for the classifier.
- ``datarun_id`` (Int): ID of the datarun associated with this classifier.
- ``hyperpartition_id`` (Int): ID of the hyperpartition associated with this
  classifier.
- ``host`` (String): IP address or name of the host machine where the classifier
  was tested.
    - not in the paper
- ``model_location`` (String): Path to the serialized model object for this
  classifier.
- ``metrics_location`` (String): Path to the full set of metrics computed during
  testing.
- ``cv_judgment_metric`` (Number): Mean of the judgement metrics from the
  cross-validated training data.
- ``cv_judgment_metric_stdev`` (Number): Standard deviation of the
  cross-validation test.
- ``test_judgment_metric`` (Number): Judgment metric computed on the test data.
- ``hyperparameters_values`` (Base64-encoded object): The full set of
  hyperparameter values used to create this classifier.
- ``start_time`` (DateTime): Time that a worker started working on the
  classifier.
- ``end_time`` (DateTime): Time that a worker finished working on the
  classifier.
- ``status`` (Enum): One of ["running", "errored", "complete"].
- ``error_message`` (String): If this classifier encountered an error, this is
  the Python stack trace from the caught exception.

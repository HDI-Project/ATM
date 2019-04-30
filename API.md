# REST API

**ATM** comes with the possibility to start a server process that enables interacting with
it via a REST API server that runs over [flask](http://flask.pocoo.org/).

In this document you will find a briefly explanation how to start it and use it.

## Starting the REST API Server

In order to start a REST API server, after installing ATM open a terminal, activate its
virtualenv, and execute this command:

```bash
atm server
```

An output similar to this one should apear in the terminal:

```bash
 * Serving Flask app "api.setup" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 150-127-826
```

After this, the REST server will be listening at the port 5000 of you machine, and if you
point your browser at http://127.0.0.1:5000/, you will see the documentation
website that shows information about all the REST operations allowed by the API.

Optionally, the `--port <port>` can be added to modify the port which the server listents at:

```bash
atm server --port 1234
```

In order to stop the server you can press <kbd>Ctrl</kbd>+<kbd>c</kbd>, but for now
you can keep it running and head to the next section.


## Quickstart

In this section we will briefly show the basic usage of the REST API.

For more detailed information about all the operations supported by the API, please point your
browser to http://127.0.0.1:5000/ and explore the examples provided by the
[Swagger](https://swagger.io/) interface.

### 1. Generate some data

Before proceeding any further, please make sure the have already populated your data by triggering
at least one model tuning process.

An easy way to do this is to follow the quickstart from the ATM [README.md](README.md) file,
which means having run these two commands:

```
atm enter_data
atm worker
```

### 2. REST Models

Once the database is populated, you can use the REST API to explore the following 4 models:

* Datasets
* Dataruns
* Hyperpartitions
* Classifiers

And these are the operations that can be performed on them:

### 3. Get all objects from a model

In order to get all the objects for a single model, you need to make a `GET` request to
`/api/<model>`.

The output will be a JSON with 4 entries:

* `num_results`: The number of results found
* `objects`: A list containing a subdocument for each result
* `page`: The current page
* `total_pages`: The number of pages

For example, you can get all the datasets using:

```
GET /api/datasets HTTP/1.1
```

And the output will be:

```
{
  "num_results": 1,
  "objects": [
    {
      "class_column": "class",
      "d_features": 16,
      "dataruns": [
        {
          "budget": 100,
          "budget_type": "classifier",
          "dataset_id": 1,
          "deadline": null,
          "description": "uniform__uniform",
          "end_time": "2019-04-11T20:58:11.346733",
          "gridding": 0,
          "id": 1,
          "k_window": 3,
          "metric": "f1",
          "priority": 1,
          "r_minimum": 2,
          "score_target": "cv_judgment_metric",
          "selector": "uniform",
          "start_time": "2019-04-11T20:58:02.514514",
          "status": "complete",
          "tuner": "uniform"
        }
      ],
      "description": null,
      "id": 1,
      "k_classes": 2,
      "majority": 0.516666667,
      "n_examples": 60,
      "name": "pollution_1",
      "size_kb": 8,
      "test_path": null,
      "train_path": "/path/to/atm/data/test/pollution_1.csv"
    }
  ],
  "page": 1,
  "total_pages": 1
}
```

### 4. Get a single object by id

In order to get one particular objects for a model, you need to make a `GET` request to
`/api/<model>/<id>`.

The output will be the document representing the corresponding object.

For example, you can get the dataset with id 1 using:

```
GET /api/datasets/1 HTTP/1.1
```

And the output will be:

```
{
  "class_column": "class",
  "d_features": 16,
  "dataruns": [
    {
      "budget": 100,
      "budget_type": "classifier",
      "dataset_id": 1,
      "deadline": null,
      "description": "uniform__uniform",
      "end_time": "2019-04-11T20:58:11.346733",
      "gridding": 0,
      "id": 1,
      "k_window": 3,
      "metric": "f1",
      "priority": 1,
      "r_minimum": 2,
      "score_target": "cv_judgment_metric",
      "selector": "uniform",
      "start_time": "2019-04-11T20:58:02.514514",
      "status": "complete",
      "tuner": "uniform"
    }
  ],
  "description": null,
  "id": 1,
  "k_classes": 2,
  "majority": 0.516666667,
  "n_examples": 60,
  "name": "pollution_1",
  "size_kb": 8,
  "test_path": null,
  "train_path": "/path/to/atm/data/test/pollution_1.csv"
}
```

### 5. Get all the children objects

In order to get all the childre objects from one parent object, you need to make a
`GET` request to `/api/<parent_model>/<parent_id>/<child_model>`.

The output will be in the same format as if you had requested all the elements from the
children model, but with the results filtered by the parent one.

So, for example, in order to get all the dataruns that use the dataset with id 1, you can use:

```
GET /api/datasets/1/dataruns HTTP/1.1
```

And the output will be (note that some parts have been cut):

```
{
  "num_results": 1,
  "objects": [
    {
      "budget": 100,
      "budget_type": "classifier",
      "classifiers": [
        {
          "cv_judgment_metric": 0.8444444444,
          "cv_judgment_metric_stdev": 0.1507184441,
          "datarun_id": 1,
          "end_time": "2019-04-11T20:58:02.600185",
          "error_message": null,
          "host": "83.56.245.36",
          "hyperparameter_values_64": "gAN9cQAoWAsAAABuX25laWdoYm9yc3EBY251bXB5LmNvcmUubXVsdGlhcnJheQpzY2FsYXIKcQJjbnVtcHkKZHR5cGUKcQNYAgAAAGk4cQRLAEsBh3EFUnEGKEsDWAEAAAA8cQdOTk5K/////0r/////SwB0cQhiQwgPAAAAAAAAAHEJhnEKUnELWAkAAABsZWFmX3NpemVxDGgCaAZDCCsAAAAAAAAAcQ2GcQ5ScQ9YBwAAAHdlaWdodHNxEFgIAAAAZGlzdGFuY2VxEVgJAAAAYWxnb3JpdGhtcRJYCQAAAGJhbGxfdHJlZXETWAYAAABtZXRyaWNxFFgJAAAAbWFuaGF0dGFucRVYBgAAAF9zY2FsZXEWiHUu",
          "hyperpartition_id": 23,
          "id": 1,
          "metrics_location": "metrics/pollution_1-4bc39b14.metric",
          "model_location": "models/pollution_1-4bc39b14.model",
          "start_time": "2019-04-11T20:58:02.539046",
          "status": "complete",
          "test_judgment_metric": 0.6250000000
        },
        ...
      ],
      "dataset": {
        "class_column": "class",
        "d_features": 16,
        "description": null,
        "id": 1,
        "k_classes": 2,
        "majority": 0.516666667,
        "n_examples": 60,
        "name": "pollution_1",
        "size_kb": 8,
        "test_path": null,
        "train_path": "/path/to/atm/data/test/pollution_1.csv"
      },
      "dataset_id": 1,
      "deadline": null,
      "description": "uniform__uniform",
      "end_time": "2019-04-11T20:58:11.346733",
      "gridding": 0,
      "hyperpartitions": [
        {
          "categorical_hyperparameters_64": "gANdcQAoWAcAAABwZW5hbHR5cQFYAgAAAGwxcQKGcQNYDQAAAGZpdF9pbnRlcmNlcHRxBIiGcQVlLg==",
          "constant_hyperparameters_64": "gANdcQAoWAwAAABjbGFzc193ZWlnaHRxAVgIAAAAYmFsYW5jZWRxAoZxA1gGAAAAX3NjYWxlcQSIhnEFZS4=",
          "datarun_id": 1,
          "id": 1,
          "method": "logreg",
          "status": "incomplete",
          "tunable_hyperparameters_64": "gANdcQAoWAEAAABDcQFjYnRiLmh5cGVyX3BhcmFtZXRlcgpGbG9hdEV4cEh5cGVyUGFyYW1ldGVyCnECY2J0Yi5oeXBlcl9wYXJhbWV0ZXIKUGFyYW1UeXBlcwpxA0sFhXEEUnEFXXEGKEc+5Pi1iONo8UdA+GoAAAAAAGWGcQeBcQh9cQkoWAwAAABfcGFyYW1fcmFuZ2VxCmgGWAUAAAByYW5nZXELXXEMKEfAFAAAAAAAAEdAFAAAAAAAAGV1YoZxDVgDAAAAdG9scQ5oAmgFXXEPKEc+5Pi1iONo8UdA+GoAAAAAAGWGcRCBcRF9cRIoaApoD2gLXXETKEfAFAAAAAAAAEdAFAAAAAAAAGV1YoZxFGUu"
        },
        ...
      ],
      "id": 1,
      "k_window": 3,
      "metric": "f1",
      "priority": 1,
      "r_minimum": 2,
      "score_target": "cv_judgment_metric",
      "selector": "uniform",
      "start_time": "2019-04-11T20:58:02.514514",
      "status": "complete",
      "tuner": "uniform"
    }
  ],
  "page": 1,
  "total_pages": 1
}
```


## Starting the REST API Server and Workers in daemon

**ATM** comes with the possibility to start a daemon process (in background) with the workers
and the REST API server. This will allow you to update dynamicly the database while new dataruns
are created for the workers.

### 1. Start the ATM process

By default **ATM** launches one worker in background if we just run the following command:

```bash
atm start
```

After starting this process, we can type:

```bash
atm status
```

And an output like this will be displayed in our console:

```
ATM is running with 1 worker
```

In order to stop this process just run:

```bash
atm stop
```

An output like this should be printed in the console:

```
ATM stopped correctly.
```

### 2. Start the ATM process with more than one worker

If we would like to launch more than one worker, we can use the argument `--workers WORKERS` or
`-w WORKERS`.

```bash
atm start -w 4
```

**Bear in mind**, if the `atm` process is allready running, a message indicating so will be
displayed when trying to start a new process.

Then if you check the `status` of `atm`:

```bash
atm status
```

The expected output is:

```
ATM is running with 4 workers
```


### 3. Start the ATM process with the REST API server

The `atm start` command accepts as an argument `--server` which will launch alongside the workers
the same REST API server as described before.

```bash
atm start --server
```

If you run `atm status` to check it's status the expected output should be as follows:

```
ATM is running with 1 worker
ATM REST server is listening on http://127.0.0.1:5000
```

### 4. Additional arguments for ATM Start

* `--sql-config SQL_CONFIG` Path to yaml SQL config file.
* `--sql-dialect {sqlite,mysql}` Dialect of SQL to use.
* `--sql-database SQL_DATABASE` Name of, or path to, SQL database.
* `--sql-username SQL_USERNAME` Username for SQL database.
* `--sql-password SQL_PASSWORD` Password for SQL database.

* `--sql-host SQL_HOST` Hostname for database machine.
* `--sql-port SQL_PORT` Port used to connect to database.

* `--sql-query SQL_QUERY` Specify extra login details.
* `--aws-config AWS_CONFIG` path to yaml AWS config file.
* `--aws-access-key AWS_ACCESS_KEY` AWS access key.
* `--aws-secret-key AWS_SECRET_KEY` AWS secret key.
* `--aws-s3-bucket AWS_S3_BUCKET` AWS S3 bucket to store data.
* `--aws-s3-folder AWS_S3_FOLDER` Folder in AWS S3 bucket in which to store data.

* `--log-config LOG_CONFIG` path to yaml logging config file.
* `--model-dir MODEL_DIR` Directory where computed models will be saved.
* `--metric-dir METRIC_DIR` Directory where model metrics will be saved.
* `--log-dir LOG_DIR`     Directory where logs will be saved.

* `--verbose-metrics` If set, compute full ROC and PR curves and per-label
metrics for each classifier.

* `--log-level-file` {critical,error,warning,info,debug,none} minimum log level to write to the
log file.

* `--log-level-stdout` {critical,error,warning,info,debug,none}
minimum log level to write to stdout.

* `--cloud-mode` Wheter to run this worker/s in cloud mode.
* `--no-save` Do not save models and metrics at all.
* `-w WORKERS` `--workers WORKERS` Number of workers.
* `--server` Also start the REST server.
* `--host HOST` IP to listen at.
* `--port PORT` Port to listen at.
* `--pid PID` PID file to use (we can use a different one in order to launch more than one process.


### 4. Stop the ATM process

As we saw before, by runing the command `atm stop` we will `terminate` the ATM process. However
this command accepts a few arguments in order to control this behaviour:

* `-t TIMEOUT`, `--timeout TIMEOUT`, time to wait in order to check if the process has been
terminated.

* `-f`, `--force`, Kill the process if it does not terminate gracefully.

### 5. Starting multiple ATM processes

**ATM** also has the posibility to launch more than one process. In order to do so, we use a `pid`
file.

By default, the `pid` file used by **ATM** is called `atm.pid`, however, you can change this name
by adding the argument `--pid` when starting **ATM**.

For example, we will start our ATM with the default values (1 worker and `atm.pid`):

```bash
atm start
```

If we run the status, this will display the following information:

```
ATM is running with 1 worker
```

Now if we would like to wake more workers we can run:

```bash
atm start --workers 4 --pid additional_workers.pid
```

In order to run the `atm status` for this `pid` add it as argument to it:

```bash
atm status --pid additional_workers.pid
```

The output of this command will be:

```
ATM is running with 4 workers
```

As you can see you will have now 5 workers running as the `SQL` configuration is the same and this
will be pointing to that database.

In order to stop the `additional_workers` process, we run `atm stop` with the `pid` file as
argument:

```bash
atm stop --pid additional_workers.pid
```

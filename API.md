# REST API

**ATM** comes with the possibility to start a server process that enables interacting with
it via a REST API server that runs over [flask](http://flask.pocoo.org/).

In this document you will find a briefly explanation how to start it and use it.


## Quickstart

In this section we will briefly show the basic usage of the REST API.

For more detailed information about all the operations supported by the API, please point your
browser to http://127.0.0.1:5000/ and explore the examples provided by the
[Swagger](https://swagger.io/) interface.


### 1. Start the REST API Server

In order to start a REST API server, after installing ATM open a terminal, activate its
virtualenv, and execute this command:

```bash
atm start
```

This will start **ATM** server as a background service. The REST server will be listening at the
port 5000 of your machine, and if you point your browser at http://127.0.0.1:5000/, you will see
the documentation website that shows information about all the REST operations allowed by the API.

Optionally, the `--port <port>` can be added to modify the port which the server listents at:

```bash
atm start --port 1234
```

If you would like to see the status of the server process you can run:

```bash
atm status
```

An output similar to this one will appear:

```bash
ATM is running with 1 worker
ATM REST server is listening on http://127.0.0.1:5000
```

In order to stop the server you can run the following command:

```bash
atm stop
```

Notice that `atm start` will start one worker by default. If you would like to launch more than one,
you can do so by adding the argument `--workers <number_of_workers` or `-w <number_of_workers>`.

```bash
atm start --workers 4
```

For more detailed options you can run `atm start --help` to obtain a list with the arguments
that are being accepted.

### 2. Create a Dataset

Once the server is running, you can register your first dataset using the API. To do so, you need
to send the path to your `CSV` file and the name of your `target_column` in a `POST` request to
`api/datasets`.

This call will create a simple `dataset` in our database:

```bash
POST /api/datasets HTTP/1.1
Content-Type: application/json

{
    "class_column": "your_target_column",
    "train_path": "path/to/your.csv"
}
```

Once you have created some datasets, you can see them by sending a `GET` request:

```bash
GET /api/datasets HTTP/1.1
```

This will return a `json` with all the information about the stored datasets.

As an example, you can get and register a demo dataset by running the following two commands:

```bash
atm get_demos
curl -v localhost:5000/api/datasets -H'Content-Type: application/json' \
-d'{"class_column": "class", "train_path": "demos/pollution_1.csv"}'
```

### 3. Trigger a Datarun

In order to trigger a datarun, once you have created a dataset, you have to send the `dataset_id`
in a `POST` request to `api/run` to trigger the `workers` with the default values.

```bash
POST /api/datasets HTTP/1.1
Content-type: application/json

{
    "dataset_id": id_of_your_dataset
}
```

If you have followed the above example and created a `pollution` dataset in the database, you can
run the following `POST` to trigger it's datarun:

```bash
curl -v localhost:5000/api/run -H'Content-type: application/json' -d'{"dataset_id": 1}'
```

**NOTE** atleast one worker should be running in order to process the datarun.

### 4. Browse the results

Once the database is populated, you can use the REST API to explore the following 4 models:

* Datasets
* Dataruns
* Hyperpartitions
* Classifiers

And these are the operations that can be performed on them:

#### Get all objects from a model

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

#### Get a single object by id

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

#### Get all the children objects

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

## Additional information

### Start additional process with different pid file

If you would like to run more workers or you would like to launch a second **ATM** process, you can
do so by specifying a different `PID` file.

For example:

```bash
atm start --no-server -w 4  --pid additional_workers.pid
```

To check the status of this process we have to run:

```bash
atm status --pid additional_workers.pid
```

This will print an output like this:

```bash
ATM is running with 4 workers
```

### Restart the ATM process

If you have an **ATM** process running and you would like to restart it and add more workers to it
or maybe change the port on which is running, you can achieve so with the `atm restart`:

```bash
atm restart
```

This command will restart the server with the default values, so if you would like to use other
options you can run `--help` to see the accepted arguments:

```bash
atm restart --help
```

### Stop the ATM process

As we saw before, by runing the command `atm stop` you will `terminate` the ATM process. However
this command accepts a few arguments in order to control this behaviour:

* `-t TIMEOUT`, `--timeout TIMEOUT`, time to wait in order to check if the process has been
terminated.

* `-f`, `--force`, Kill the process if it does not terminate gracefully.
* `--pid PIDFILE`, PID file to use

### Start the ATM REST API server in foreground

If you would like to monitorize the server for debugging process, you can do so by runing the
with the following command:

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

For additional arguments run `atm server --help`

**Note** that this command will not launch any `workers` process. In order to launch a foreground
worker you have to do so by runing `atm worker`.

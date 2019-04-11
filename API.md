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

If you would like to start the server in another port, which by default it's 5000, you can include
the `--port` option to run it at the port that you would like:

```bash
atm server --port 1234
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

If you now point your browser at http://127.0.0.1:5000/, you will see the documentation
website that shows information about all the REST operations allowed by the API.

You can press <kbd>Ctrl</kbd>+<kbd>c</kbd> at any moment to stop the process, but for now
you can keep it running and head to the next section.


## Usage

For this example we have run `atm enter_data` with the default dataset and `atm worker` in order
to create the classifiers and to populate our database.

By accessing the http://127.0.0.1:5000/ you will see the [Swagger](https://swagger.io/)
documentation and be able to run examples and calls to the REST API.

In the following steps we will explain how to use this **API**:

**ATM** REST API allows you to navigate arround the database and have access the following tables:

* Dataset
* Datarun
* Hyperpartition
* Classifier

### Dataset

In order to retrieve the information stored for a `Dataset`, the available parameters to create
an API call are as follow:

* class_column (string, optional)
* d_features (integer, optional)
* dataruns (Array[Datarun], optional)
* description (string, optional)
* id (integer, optional)
* k_classes (integer, optional)
* majority (number, optional)
* n_examples (integer, optional)
* name (string, optional)
* size_kb (integer, optional)
* test_path (string, optional)
* train_path (string, optional)

If you are using `Unix` and you have [CURL](https://curl.haxx.se/) you can run this commands in
a separate terminal, otherwise you can access and visualize the data recived from your browser.

```bash
curl -X GET --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/datasets'
```

This should print an output out of the database similar to this one:

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
          "end_time": "2019-04-11T17:26:28.781095",
          "gridding": 0,
          "id": 1,
          "k_window": 3,
          "metric": "f1",
          "priority": 1,
          "r_minimum": 2,
          "score_target": "cv_judgment_metric",
          "selector": "uniform",
          "start_time": "2019-04-11T17:25:57.192200",
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
      "train_path": "/test/pollution_1.csv"
    }
  ],
  "page": 1,
  "total_pages": 1
}

```

If you would like to recover a certain dataset, we can do so by `id`:

```bash
curl -X GET "http://127.0.0.1:5000/api/datasets/10" -H "accept: application/json"
```

Where `10` is the `id` of our dataset.
If you have the database created from our example, containing only one dataset, the output to this
call should be empty:

```bash
{}
```

If you would like to delete a dataset, you need it's `id` and run:

```bash
curl -X DELETE --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/datasets/16'
```

Where `16` is the `id` of the dataset.

### Datarun

* budget (integer, optional),
* budget_type (string, optional),
* classifiers (Array[Classifier], optional),
* dataset (Dataset, optional),
* dataset_id (integer, optional),
* deadline (string, optional),
* description (string, optional),
* end_time (string, optional),
* gridding (integer, optional),
* hyperpartitions (Array[Hyperpartition], optional),
* id (integer, optional),
* k_window (integer, optional),
* metric (string, optional),
* priority (integer, optional),
* r_minimum (integer, optional),
* score_target (string, optional),
* selector (string, optional),
* start_time (string, optional),
* status (string, optional),
* tuner (string, optional)

```bash
curl -X GET --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/dataruns'
```

This should print an output out of the database similar to this one:

```
{
  "num_results": 1,
  "objects": [
    {
      "budget": 100,
      "budget_type": "classifier",
      "classifiers": [
        {
          "cv_judgment_metric": 0.7120634921,
          "cv_judgment_metric_stdev": 0.1153100042,
          "datarun_id": 1,
          "end_time": "2019-04-11T17:25:57.412273",
          "error_message": null,
          "host": "83.56.245.36",
          "hyperparameter_values_64": "gAN9cQAoWAsAAABuX25laWdoYm9yc3EBY251bXB5LmNvcmUubXVsdGlhcnJheQpzY2FsYXIKcQJjbnVtcHkKZHR5cGUKcQNYAgAAAGk4cQRLAEsBh3EFUnEGKEsDWAEAAAA8cQdOTk5K/////0r/////SwB0cQhiQwgSAAAAAAAAAHEJhnEKUnELWAcAAAB3ZWlnaHRzcQxYCAAAAGRpc3RhbmNlcQ1YCQAAAGFsZ29yaXRobXEOWAUAAABicnV0ZXEPWAYAAABtZXRyaWNxEFgJAAAAbWFuaGF0dGFucRFYBgAAAF9zY2FsZXESiHUu",
          "hyperpartition_id": 31,
          "id": 1,
          "metrics_location": "metrics/pollution_1-fd916442.metric",
          "model_location": "models/pollution_1-fd916442.model",
          "start_time": "2019-04-11T17:25:57.273278",
          "status": "complete",
          "test_judgment_metric": 0.9523809524
        },
...
```

If you would like to recover a certain datarun, we can do so by `id`:

```bash
curl -X GET "http://127.0.0.1:5000/api/dataruns/10" -H "accept: application/json"
```

Where `10` is the `id` of our dataset.
If you have the database created from our example, containing only one dataset, the output to this
call should be empty:

```bash
{}
```

If you would like to delete a datarun, you need it's `id` and run:

```bash
curl -X DELETE --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/dataruns/16'
```

Where `16` is the `id` of the datarun.

### Hyperpartition

* categorical_hyperparameters_64 (string, optional),
* classifiers (Array[Classifier], optional),
* constant_hyperparameters_64 (string, optional),
* datarun (Datarun, optional),
* datarun_id (integer, optional),
* id (integer, optional),
* method (string, optional),
* status (string, optional),
* tunable_hyperparameters_64 (string, optional)

```bash
curl -X GET --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/hyperpartitions'
```

This should print an output out of the database similar to this one:

```
{
  "num_results": 32,
  "objects": [
    {
      "categorical_hyperparameters_64": "gANdcQAoWAcAAABwZW5hbHR5cQFYAgAAAGwxcQKGcQNYDQAAAGZpdF9pbnRlcmNlcHRxBIiGcQVlLg==",
      "classifiers": [
        {
          "cv_judgment_metric": 0E-10,
          "cv_judgment_metric_stdev": 0E-10,
          "datarun_id": 1,
          "end_time": "2019-04-11T17:25:58.591654",
          "error_message": null,
          "host": "83.56.245.36",
          "hyperparameter_values_64": "gAN9cQAoWAEAAABDcQFjbnVtcHkuY29yZS5tdWx0aWFycmF5CnNjYWxhcgpxAmNudW1weQpkdHlwZQpxA1gCAAAAZjhxBEsASwGHcQVScQYoSwNYAQAAADxxB05OTkr/////Sv////9LAHRxCGJDCJx3VDODxC8/cQmGcQpScQtYAwAAAHRvbHEMaAJoBkMIFQYn8/JBj0BxDYZxDlJxD1gHAAAAcGVuYWx0eXEQWAIAAABsMXERWA0AAABmaXRfaW50ZXJjZXB0cRKIWAwAAABjbGFzc193ZWlnaHRxE1gIAAAAYmFsYW5jZWRxFFgGAAAAX3NjYWxlcRWIdS4=",
          "hyperpartition_id": 1,
          "id": 7,
          "metrics_location": "metrics/pollution_1-b2ac0bd8.metric",
          "model_location": "models/pollution_1-b2ac0bd8.model",
          "start_time": "2019-04-11T17:25:58.476363",
          "status": "complete",
          "test_judgment_metric": 0E-10
        },
...
```

If you would like to recover a certain hyperpartition, we can do so by `id`:

```bash
curl -X GET "http://127.0.0.1:5000/api/hyperpartition/10" -H "accept: application/json"
```

Where `10` is the `id` of our hyperpartition.

If you have the database created from our example, containing only one dataset, the output to this
call should be empty:

```bash
{}
```

If you would like to delete a hyperpartition, you need it's `id` and run:

```bash
curl -X DELETE --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/hyperpartitions/16'
```

Where `16` is the `id` of the hyperpartition.

### Classifier

* cv_judgment_metric (number, optional),
* cv_judgment_metric_stdev (number, optional),
* datarun (Datarun, optional),
* datarun_id (integer, optional),
* end_time (string, optional),
* error_message (string, optional),
* host (string, optional),
* hyperparameter_values_64 (string, optional),
* hyperpartition (Hyperpartition, optional),
* hyperpartition_id (integer, optional),
* id (integer, optional),
* metrics_location (string, optional),
* model_location (string, optional),
* start_time (string, optional),
* status (string, optional),
* test_judgment_metric (number, optional)

```bash
curl -X GET --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/classifiers'
```

This should print an output out of the database similar to this one:

```
{
  "num_results": 100,
  "objects": [
    {
      "cv_judgment_metric": 0.7120634921,
      "cv_judgment_metric_stdev": 0.1153100042,
      "datarun": {
        "budget": 100,
        "budget_type": "classifier",
        "dataset_id": 1,
        "deadline": null,
        "description": "uniform__uniform",
        "end_time": "2019-04-11T17:26:28.781095",
        "gridding": 0,
        "id": 1,
        "k_window": 3,
        "metric": "f1",
        "priority": 1,
        "r_minimum": 2,
        "score_target": "cv_judgment_metric",
        "selector": "uniform",
        "start_time": "2019-04-11T17:25:57.192200",
        "status": "complete",
        "tuner": "uniform"
      },
      "datarun_id": 1,
      "end_time": "2019-04-11T17:25:57.412273",
      "error_message": null,
      "host": "83.56.245.36",
      "hyperparameter_values_64": "gAN9cQAoWAsAAABuX25laWdoYm9yc3EBY251bXB5LmNvcmUubXVsdGlhcnJheQpzY2FsYXIKcQJjbnVtcHkKZHR5cGUKcQNYAgAAAGk4cQRLAEsBh3EFUnEGKEsDWAEAAAA8cQdOTk5K/////0r/////SwB0cQhiQwgSAAAAAAAAAHEJhnEKUnELWAcAAAB3ZWlnaHRzcQxYCAAAAGRpc3RhbmNlcQ1YCQAAAGFsZ29yaXRobXEOWAUAAABicnV0ZXEPWAYAAABtZXRyaWNxEFgJAAAAbWFuaGF0dGFucRFYBgAAAF9zY2FsZXESiHUu",
      "hyperpartition": {
        "categorical_hyperparameters_64": "gANdcQAoWAcAAAB3ZWlnaHRzcQFYCAAAAGRpc3RhbmNlcQKGcQNYCQAAAGFsZ29yaXRobXEEWAUAAABicnV0ZXEFhnEGWAYAAABtZXRyaWNxB1gJAAAAbWFuaGF0dGFucQiGcQllLg==",
        "constant_hyperparameters_64": "gANdcQBYBgAAAF9zY2FsZXEBiIZxAmEu",
        "datarun_id": 1,
        "id": 31,
        "method": "knn",
        "status": "incomplete",
        "tunable_hyperparameters_64": "gANdcQBYCwAAAG5fbmVpZ2hib3JzcQFjYnRiLmh5cGVyX3BhcmFtZXRlcgpJbnRIeXBlclBhcmFtZXRlcgpxAmNidGIuaHlwZXJfcGFyYW1ldGVyClBhcmFtVHlwZXMKcQNLAYVxBFJxBV1xBihLAUsUZYZxB4FxCH1xCShYDAAAAF9wYXJhbV9yYW5nZXEKaAZYBQAAAHJhbmdlcQtdcQwoSwFLFGV1YoZxDWEu"
      },
```

If you would like to recover a certain classifier, we can do so by `id`:

```bash
curl -X GET "http://127.0.0.1:5000/api/classifiers/10" -H "accept: application/json"
```

Where `10` is the `id` of our classifier.

If you have the database created from our example, containing only one dataset, the output to this
call should be empty:

```bash
{}
```

If you would like to delete a classifiers, you need it's `id` and run:

```bash
curl -X DELETE --header 'Accept: application/json, application/json' 'http://127.0.0.1:5000/api/classifiers/16'
```

Where `16` is the `id` of the classifiers.

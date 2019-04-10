# REST API

**ATM** comes the possibility to start a server process that enables interacting with
it via a REST API server that runs over [flask](http://flask.pocoo.org/).

In this document we briefly explain how to start it.

## Starting the REST API Server

In order to start a REST API server, after installing ATM open a terminal, activate its
virtualenv, and execute this command:

```bash
python scripts/rest_api_server.py
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

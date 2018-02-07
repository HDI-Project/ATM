Setup
=====

1. Package Installation
-----------------------

`Python <https://www.python.org>`_, `pip <https://pip.pypa.io/en/stable>`_, and `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ must be installed manually.
All other packages can be installed automatically with ``pip``.
After installing ``Python`` and ``pip``, ``virtualenv`` can be installed by::

    $ sudo apt-get install python-pip
    $ sudo pip install virtualenv


Virtual Environment Setup
^^^^^^^^^^^^^^^^^^^^^^^^^

Create the virtual environment and enter into it::

    $ virtualenv atm-env
    (atm-env) $

Install the required packages::

    (atm-env) $ pip install -r requirements.txt

The required packages are:

.. literalinclude:: ../../requirements.txt


2. Database Setup
-----------------

On Ubuntu, SQLite3 can be installed with the command::

    $ sudo apt-get install sqlite3

Alternatively, MySQL can be installed with the command::

    $ sudo apt-get install mysql-server mysql-client libmysqlclient-dev

and following the instructions.

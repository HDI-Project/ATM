Setup
=====
This page will guide you though downloading and installing ATM.

0. Requirements
---------------

Currently, ATM is only compatible with Python 2.7, 3.5 and 3.6 and \*NIX systems.

We also recommend using `virtualenv <https://virtualenv.pypa.io/en/stable/>`_, which
you can install as follows.::

    $ sudo apt-get install python-pip
    $ sudo pip install virtualenv

For development, also `git <https://git-scm.com/>`_ is required in order to download and
update the software.

1. Install ATM
--------------

Install using pip
~~~~~~~~~~~~~~~~~

The recommended way to install ATM is using `pip <https://pip.pypa.io/en/stable>`_ inside
a dedicated virtualenv::

    $ virtualenv atm-env
    $ . atm-env/bin/activate
    (atm-env) $ pip install atm

Install from source
~~~~~~~~~~~~~~~~~~~

Alternatively, and for development, you can clone the repository and install it from
source by running ``make install``::

    $ git clone https://github.com/hdi-project/atm.git
    $ cd atm
    $ virtualenv atm-env
    $ . atm-env/bin/activate
    (atm-env) $ make install

For development, replace the last command with ``make install-develop`` command in order to
also install all the required dependencies for testing and linting.

.. note:: You will need to execute the command ``. atm-env/bin/activate`` to activate the
          virtualenv again every time you want to start working on ATM. You will know that your
          virtualenv has been activated if you can see the **(atm-env)** prefix on your prompt.
          If you do not, activate it again!

2. Install a database
---------------------

ATM requires a SQL-like database to store information about datasets, dataruns,
and classifiers. It's currently compatible with the SQLite3 and MySQL dialects.
For first-time and casual users, we recommend installing SQLite::

    $ sudo apt-get install sqlite3

If you're planning on running large, distributed, or performance-intensive jobs,
you might prefer using MySQL. Run::

    $ sudo apt-get install mysql-server mysql-client

and following the instructions.

3. Start using ATM!
-------------------

You're all set. Head over to the `quick-start <quickstart.html>`_ section to create and
execute your first job with ATM.

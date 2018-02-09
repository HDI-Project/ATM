Setup
=====
This page will guide you though downloading and installing ATM.

0. Requirements
---------------
Currently, ATM is only compatible with Python 2.7 and \*NIX systems, and `git
<https://git-scm.com/>`_ is required to download and update the software.

1. Clone the project
-----------------------
From the terminal, run::

    $ git clone https://github.com/hdi-project/atm.git ./atm


2. Install a database
-----------------------

ATM requires a SQL-like database to store information about datasets, dataruns,
and classifiers. It's currently compatible with the SQLite3 and MySQL dialects.
For first-time and casual users, we recommend installing SQLite::

    $ sudo apt-get install sqlite3

If you're planning on running large, distributed, or performance-intensive jobs,
you might prefer using MySQL. Run::

    $ sudo apt-get install mysql-server mysql-client

and following the instructions.

No matter which you choose, you'll need to install the mysql client developer
library in order for SQLAlchemy to work correctly::

    $ sudo apt-get install libmysqlclient-dev

3. Install Python dependencies
------------------------------

We recommend using `pip <https://pip.pypa.io/en/stable>`_ and `virtualenv
<https://virtualenv.pypa.io/en/stable/>`_ to make this process easier.::

    $ sudo apt-get install python-pip
    $ sudo pip install virtualenv

Next, create the virtual environment and enter into it::

    $ virtualenv atm-env
    $ . atm-env/bin/activate
    (atm-env) $

The required packages are:

.. literalinclude:: ../../requirements.txt

Install them with pip::

    (atm-env) $ pip install -r requirements.txt

Or, if you want to use ATM as a library in other applications, you can install
it as a package. This will install the requirements as well::

    (atm-env) $ pip install -e . --process-dependency-links

You're all set. Head over to the `tutorial <tutorial>`_ section to create and
execute your first job with ATM.

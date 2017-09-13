Setup
=====

1. Package Installation
-----------------------

`Python <https://www.python.org>`_, `pip <https://pip.pypa.io/en/stable>`_, and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ must be installed manually.
All other packages can be installed automatically with ``virtualenvwrapper``.
After installing ``Python`` and ``pip``, ``virtualenvwrapper`` can be installed by::

    $ sudo apt-get install python-pip
    $ sudo pip install virtualenv
    $ sudo pip install virtualenvwrapper

Add into ``~/.bash_profile``::

    # for vitrualenvwrapper
    export WORKON_HOME=$HOME/.virtualenvs
    source /usr/local/bin/virtualenvwrapper_lazy.sh


Virtual Environment Setup
^^^^^^^^^^^^^^^^^^^^^^^^^

Create the virtual environment and enter into it::

    $ mkvirtualenv atm-env
    $ workon atm-env
    (atm-env) $

Install the required packages::

    (atm-env) $ pip install -r setup/reqs.txt

The required packages are:

.. literalinclude:: ../../setup/reqs.txt


2. MySQL Setup
--------------

On Ubuntu, MySQL can be installed with the command::

    $ sudo apt-get install git python-dev mysql-server mysql-client gfortran libatlas-base-dev libmysqlclient-dev build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base libfreetype6-dev libxft-dev libssl-dev

and following the instructions.
Once MySQL is installed, setup the ATM DataHub database::

    $ mysql
    mysql> create database atm_db;
    mysql> exit;
    $ mysql atm_db < setup/hyperatm.sql

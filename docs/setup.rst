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

2. Install a database (Optional)
--------------------------------

ATM uses a database to store information about datasets, dataruns and classifiers.
It's currently compatible with the SQLite3 and MySQL dialects.

For first-time and casual users, the SQLite3 is used by default without any required
step from the user.

However, if you're planning on running large, distributed, or performance-intensive jobs,
you might prefer using MySQL.

If you do not have a MySQL database already prepared, you can follow the next steps in order
install it and parepare it for ATM:

1. **Install mysql-server**

First install mysql-server using the following command::

    sudo apt-get install mysql-server

In the latest versions of MySQL no input for the user is required for this step, but
in older versions the installation process will require the user to input a password
for the MySQL root user.

If this happens, keep track of the password that you set, as you will need it in the
next step.

2. **Log into your MySQL instance as root**

If no password was required during the installation of MySQL, you should be able to
log in with the following command::

    sudo mysql

If a MySQL Root password was required, you will need to execute this other command::

    sudo mysql -u root -p

and input the password that you used during the installation when prompted.

3. **Create a new Database for ATM**

Once you are logged in, execute the following three commands to create a database
called `atm` and a user also called `atm` with write permissions on it::

    $ mysql> CREATE DATABASE atm;
    $ mysql> CREATE USER 'atm'@'localhost' IDENTIFIED BY 'set-your-own-password-here';
    $ mysql> GRANT ALL PRIVILEGES ON atm.* TO 'atm'@'localhost';

4. **Test your settings**

After you have executed the previous three commands and exited the mysql prompt,
you can test your settings by executing the following command and inputing the
password that you used in the previous step when prompted::

    mysql -u atm -p

3. Start using ATM!
-------------------

You're all set. Head over to the `quick-start <quickstart.html>`_ section to create and
execute your first job with ATM.

BTB - Bayesian Tuning and Building
====

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://hdi-project.github.io/BTB/)

Install dependencies:
$ virtualenv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
$ sudo apt install mysql-server mysql-client

Set up MySQL:
$ mysql -u root -p
> GRANT ALL ON btb.\* TO 'username'@'localhost' IDENTIFIED BY 'password';
> CREATE DATABASE btb;

$ mysql -u username -p btb < setup/hyperbtb.sql
You will be prompted for your password. This will set up the BTB database

$ mysql -u username -p btb
> show tables;
you should see:
+---------------+
| Tables_in_btb |
+---------------+
| algorithms    |
| dataruns      |
| frozen_sets   |
| learners      |
+---------------+
4 rows in set (0.00 sec)

Edit btb.cnf to add your settings.

Create a datarun 
$ python btb/enter_data.py --configpath ./path/to/btb.cnf 

Note: Any dataset with less than 30 samples will fail for the DBN classifier unless the DBN `minibatch_size` constant is changed to match the number of samples.

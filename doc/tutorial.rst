Tutorial
========

Data Format
-----------
Delphi takes data in CSV format where each CSV file needs to:

* Have the first line of the file be headers with strings as the feature names, and the class column named "class". If the features aren't named (ie, image or SVD or PCA data), then anything will do (but see below for a small script to generate nice feature names).
* Should have N + 1 rows (1 header + N examples)
* Should have D + 1 features (1 class label + D features per example)

Here's a handy Python script to create a CSV header line for data that doesn't have feaure names::


    def CreateCSVHeader(n_features, name, class_label_name):
        """
            Creates a CSV header like:
                "<class_label_name>, <name>1, <name>2, ..., <name><n_features>"

            Example:
                print CreateCSVHeader(64, "pixel", "class")
        """
        separator = ","
        header_row_string = separator.join(
            [x + str(y) for (x, y) in
                zip([name for i in range(n_features)], range(1, n_features + 1, 1))])
        return separator.join([class_label_name, header_row_string])


Creating a DataRun in DataHub
-----------------------------

Once your data in the proper format, you can upload it to the DataHub for processing.

Run Mode
^^^^^^^^

You must decide which mode you want to Delphi in: local or cloud (see :doc:`intro` for more info).
Delphi defaults to cloud mode, but you can activate local mode by creating the following environmental variable::

    (delphi-env) $ export DELPHI_ENV=LOCAL

Configuration File
^^^^^^^^^^^^^^^^^^
    
To run Delphi, you must create a configuration file.
A configuration file template is included in ``config/delphi.cnf.template`` (and shown below).
Since the configuration file contains passwords, it's best to rename it to ``delphi.cnf`` so that it will be ignored by git.
This is especially true if you plan to make changes to Delphi and upload them to the repository. 
The git repository is setup to ignore all files in the ``config`` folder except ``delphi.cnf.template``.

.. literalinclude:: ../config/delphi.cnf.template

DataRun Creation
^^^^^^^^^^^^^^^^

Now we need to add the `datarun` to the DataHub (database).
A datarun consists of all the parameters for a single experiment run, including where the find the data, what the budget is for number of learners to train, the majoirty class benchmark, and other things.
The datarun ID in the database also ties together the `hyperpartitions` (frozen sets) which delinate how Delphi can explore different subtypes of classifiers to maximize their performance.
Once the configuration file is filled out, we can enter it in DataHub with::

    (delphi-env) $ python run_fromfile.py
    
Note: For both local or cloud mode, the database connection details must be entered in ``delphi/databases.py``.

Workers
-------

Once at least one datarun is in the DataHub, workers can be started to run classification routines.

Local Mode
^^^^^^^^^^

In local mode, this is simple::
    
    (delphi-env) $ python run_fromfile.py
    
This command can b executed several times to create many workers that operate independently in parallel.
How many to run depends of your judgement of your computer's capabilities.

Cloud Mode
^^^^^^^^^^
In cloud mode, the `fabric <https://www.fabfile.org>`_ package is used to deploy workers on AWS.
First, you must create instance(s) by executing the following command on your local machine::

    (delphi-env) $ fab create_instances
    
The number of instances will be the number specified in the *aws* section of the configuration file.
Second, you must start Delphi worker processes on the instance(s) by executing the following command on your local machine::

    (delphi-env) $ fab deploy
    
The number of worker processes on each EC2 instance is specified in the configuration file.
Once the workers are done, the worker processes can be killed by executing the following command on your local machine::

    (delphi-env) $ fab killworkers

**This does NOT terminate the instances**.
Currently, the instances have to be terminated from the AWS Management Console.


Note: For both local or cloud mode, the database connection details must be entered in ``delphi/databases.py``.
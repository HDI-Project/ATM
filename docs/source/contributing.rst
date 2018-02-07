Extending and contributing to ATM
=================================

Adding a classification method
------------------------------

ATM includes several classification methods out of the box, but it's possible to add new methods as well.

The method must be implemented by a python class that has the following instance methods:

1) ``fit``: accepts training data and labels (X and y) and trains a predictive model.
2) ``predict``: accepts a matrix of unlabeled feature vectors (X) and returns predictions for the corresponding labels (y).

This follows the convention used by `scikit-learn <http://scikit-learn.org/stable/>`_, and most of the classifier methods already included with ATM are ``sklearn`` classes. However, any custom python class that implements the fit/predict interface can be used with ATM.

Once you have a class, you need to configure the relevant hyperparameters and tell ATM about your class.

1. Creating the JSON file
^^^^^^^^^^^^^^^^^^^^^^^^^

All configuration for a classification method must be described in a json file with the following format:

.. code-block:: javascript

    {
        "name": "bnb",
        "class": "sklearn.naive_bayes.BernoulliNB",
        "parameters": {...},
        "root_parameters": [...],
        "conditions": {...}
    }

"name" is a short string (or "code") which ATM uses to refer to the method.
"class" is an import path to the class which Python can interpret.
"parameters" is a list of hyperparameters which ATM will attempt to tune. Each parameter has two fields: "type" and "range". The type is one of ["float", "float_exp", "float_cat", "int", "int_exp", "int_cat", "string", "bool"]. Types ending in "_cat" are categorical types, and those ending in "_exp" are exponential types. The range is a list of values which define the range of the variable.

For categorical types ("bool", "string", "int_cat", and "float_cat"), "range" is an explicit list of all possible values. For non-categorical numeric types, the range defines lower and upper bounds, respectively, for the parameter. Numeric ranges are inclusive (i.e. [0.0, 1.0] includes both 0.0 and 1.0).

Example categorical types:

.. code-block:: javascript

    "nu": {
        "type": "float_cat",
        "range": [0.5, 1.5, 2.5]  // will select one of the listed values
    }

    "kernel": {
        "type": "string",
        "range": ["constant", "rbf", "matern"]  // will select one of the listed strings
    }

Example numeric type:

.. code-block:: javascript

    "max_depth": {
        "type": "int",
        "range": [2, 10]   // will select integer values uniformly at random between 2 and 10, inclusive
    }

Example exponential type:

.. code-block:: javascript

    "length_scale": {
        "type": "float_exp",
        "range": [1e-5, 1e5]  // will select floating-point values from an exponential distribution between 10^-5 and 10^5, inclusive
    }


There are two kinds of hyperparameters: root parameters (also referred to as method parameters in the paper) and conditional parameters. Root parameters must be passed to the method class's constructor no matter what, and conditional parameters must only be passed if specific values for other parameters are set. For example, the GaussianProcessClassifier configuration has a single root parameter: ``kernel``. This must be set no matter what. Depending on how it's set, other parameters might need to be set as well. The format for conditions is as follows:

.. code-block:: javascript

    {
        "root_parameter": {
            "root_param_value": ["conditional_parameters"]
        }
    }

In ``gaussian_process.json``, there are three sets of parameters which are conditioned on the value of the root parameter ``kernel``:

.. code-block:: javascript

    "root_parameters": ["kernel"],

    "conditions": {
        "kernel": {
            "matern": ["nu"],
            "rational_quadratic": ["length_scale", "alpha"],
            "exp_sine_squared": ["length_scale", "periodicity"]
        }
    }


If ``kernel`` is set to "matern", it means ``nu`` must also be set. If it's set to "rational_quadratic" instead, ``length_scale`` and ``alpha`` must be set instead. Conditions can overlap -- for instance, ``length_scale`` must be set if kernel is either "rational_quadratic" or "exp_sine_squared", so it's included in both conditional lists. The only constraint is that any parameter which is set as a result of a condition (i.e. a conditional parameter) must not be listed in "root_parameters".

2. (Optional) Adding a method to the ATM library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We welcome contributions of new methods! If your method is implemented as part of a publicly-available Python library which is compatible with ATM’s other dependencies, you can submit it for permanent inclusion in the library.

Save a copy of your configuration json in the ``methods/`` directory. Then, in ``atm/constants.py``, enter a mapping from the method's name to the name of its json file in the ``METHODS_MAP`` dictionary. If necessary, add the library where your method lives to ``requirements.txt``.

Test out your method with ``python test/method_test.py --method <your method code>``.  If all hyperpartitions run error-free, you're probably good to go. Commit your changes to a separate branch, then open up a pull request in the main repository. Explain why your method is a useful addition to ATM, and we'll incorporate it if we agree!


Adding a new hyperpartition Selector
------------------------------------
A parameter selector can be created by creating a class which inherits the ``btb.Selector`` class. The class must have a ``select`` function which returns the chose parameters. 


Changing the Acquisition Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Gaussian Process Expected Improvement selection scheme makes use of an acquisition function to decide which parameter set will offer the best performance improvement.  The current acquisition function (seen below) uses the predicted performance and the confidence in that prediction to decide which hyperpartition to try next. This metric can be altered depending on the needs of a particular problem by modifying.


Adding a new hyperparameter Tuner
---------------------------------
A parameter selector can be created by creating a class which inherits the ``btb.Tuner`` class. The class must have a ``select`` function which returns the chose parameters.  An example which uses the UCB1 algorithm to choose the hyperpartition is shown below.


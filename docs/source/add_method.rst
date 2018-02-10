Adding a classification method
==============================

ATM includes several classification methods out of the box, but it's possible to
add custom ones too.

From 10,000 feet, a "method" in ATM comprises the following:

1. A Python class which defines a fit-predict interface;

2. A set of *hyperparameters* that are (or may be) passed to the class's
   constructor, and the range of values that each hyperparameter may take;

3. A *conditional parameter tree* that defines how hyperparameters depend on one
   another; and
   
4. A JSON file in ``atm/methods/`` that describes all of the above.

1. Valid method classes
-------------------------
Every method must be implemented by a python class that has the following
instance methods:

1) ``fit``: accepts training data and labels (X and y) and trains a predictive model.
2) ``predict``: accepts a matrix of unlabeled feature vectors (X) and returns predictions for the corresponding labels (y).

This follows the convention used by `scikit-learn <http://scikit-learn.org/stable/>`_, and most of the classifier methods already included with ATM are ``sklearn`` classes. However, any custom python class that implements the fit/predict interface can be used with ATM.

Once you have a class, you need to configure the relevant hyperparameters and tell ATM about your class.

2. Creating the JSON file
-------------------------
All configuration for a classification method must be described in a json file with the following format:

.. code-block:: javascript

    {
        "name": "bnb",
        "class": "sklearn.naive_bayes.BernoulliNB",
        "hyperparameters": {...},
        "root_hyperparameters": [...],
        "conditions": {...}
    }

- "name" is a short string (or "code") which ATM uses to refer to the method.
- "class" is an import path to the class which Python can interpret.
- "hyperparameters" is a list of hyperparameters which ATM will attempt to tune. 
  
Defining hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^
Most parameter definitions have two fields: "type" and either "range" or "values". 
The "type" is one of ["float", "float_exp", "float_cat", "int", "int_exp",
"int_cat", "string", "bool"]. Types ending in "_cat" are categorical
types, and those ending in "_exp" are exponential types. 

- If the type is ordinal or continuous (e.g. "int" or "float"), "range"
  defines the upper and lower bound on possible values for the parameter.
  Ranges are inclusive: [0.0, 1.0] includes both 0.0 and 1.0.
- If the type is categorical (e.g. "string" or "float_cat"), "values"
  defines the list of all possible values for the parameter.

Example categorical types:

.. code-block:: javascript

    "nu": {
        "type": "float_cat",
        "values": [0.5, 1.5, 3.5]  // will select one of the listed values
    }

    "kernel": {
        "type": "string",
        "values": ["constant", "rbf", "matern"]  // will select one of the listed strings
    }

Example (uniform) numeric type:

.. code-block:: javascript

    "max_depth": {
        "type": "int",
        "range": [2, 10]   // will select integer values uniformly at random between 2 and 10, inclusive
    }

Example exponential numeric type:

.. code-block:: javascript

    "length_scale": {
        "type": "float_exp",
        "range": [1e-5, 1e5]  // will select floating-point values from an exponential distribution between 10^-5 and 10^5, inclusive
    }


Defining the Conditional Parameter Tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are two kinds of hyperparameters: *root hyperparameters* (also referred to
as "method hyperparameters" in the paper) and *conditional parameters*. Root parameters
must be passed to the method class's constructor no matter what, and conditional
parameters are only passed if specific values for other parameters are set.  For
example, the GaussianProcessClassifier configuration has a single root
parameter: ``kernel``. This must be set no matter what. Depending on how it's
set, other parameters might need to be set as well. The format for conditions is
as follows:

.. code-block:: javascript

    {
        "root_parameter_name": {
            "value1": ["conditional_parameter_name", ...],
            "value2": ["other_conditional_parameter_name", ...]
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

The example above defines a conditional parameter tree that looks something like
this::
    kernel-----------------------  
    |        \                   \ 
    matern    rational_quadratic  exp_sine_squared
    |         |           |       |             |    
    nu      length_scale  alpha   length_scale  periodicity 


3. (Optional) Adding a new method to the ATM library
----------------------------------------------------
We are always looking for new methods to add to ATM's core! If your method is
implemented as part of a publicly-available Python library which is compatible
with ATM’s other dependencies, you can submit it for permanent inclusion in the
library.

Save a copy of your configuration json in the ``atm/methods/`` directory. Then, in
in the ``METHODS_MAP`` dictionary in ``atm/constants.py``, enter a mapping from
a short string representing your method's name to the name of its json file. For
example, ``'dt': 'decision_tree.json'``. If necessary, add the library where
your method lives to ``requirements.txt``.

Test out your method with ``python scripts/test_method.py --method
<your_method_code>``.  If all hyperpartitions run error-free, you're probably
good to go.  Commit your changes to a separate branch, then open up a pull
request in the main repository. Explain why your method is a useful addition to
ATM, and we'll merge it in if we agree!


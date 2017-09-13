atm
======

Adding a classifier
-------------------

The classifier must be a python class that has the following functions:

1) ``fit``: given training data and labels, learns the model
2) ``predict``: given data sample(s), predicts the label(s)


In ``atm/mapping.py``, enter the function in the ``LEARNER_CODE_CLASS_MAP`` variable and the classifier enumerator (see below) in the ``ENUMERATOR_CODE_CLASS_MAP`` variable.
Decide a code for the classifier (e.g., classify_knn) and enter this in the ``__init__.py`` file in ``atm/enumeration/classification``.

Create an Enumerator for the classifier in the ``atm/enumeration/classification`` folder.
In this file:

a) define the ranges of the parameters in a variable called ``DEFAULT_RANGES``
b) define the parameter types (int, float, etc.) in a variable called ``DEFAULT_KEYS``
c) define the Conditional Parameter Tree (CPT) in a function called ``create_cpt``

.. code:: python

  from atm.cpt import Choice, Combination
  from atm.enumeration import Enumerator
  from atm.enumeration.classification import ClassifierEnumerator
  from atm.key import Key, KeyStruct
  import numpy as np

  class EnumeratorExample(ClassifierEnumerator):

      DEFAULT_RANGES = {
          "param_int" : (a, b),
          "param_categorical" : ('category1', 'category2'),
          "param_float" : (c, d),
      }

      DEFAULT_KEYS = {
          # KeyStruct(range, key_type, is_categorical)
          "param_int" : KeyStruct(DEFAULT_RANGES["param_int"], Key.TYPE_INT, False),
          "param_categorical" : KeyStruct(DEFAULT_RANGES["param_categorical"], Key.TYPE_STRING, True),
          "param_float" : KeyStruct(DEFAULT_RANGES["param_float"], Key.TYPE_FLOAT, False),
      }

      def __init__(self, ranges=None, keys=None):
          super(EnumeratorExample, self).__init__(
              ranges or EnumeratorExample.DEFAULT_RANGES, keys or EnumeratorExample.DEFAULT_KEYS)
          self.code = ClassifierEnumerator.Example
          self.create_cpt()

      def create_cpt(self):
          param_int = Choice("param_int", self.ranges["param_int"])
          param_categorical = Choice("param_categorical", self.ranges["param_categorical"])
          param_float = Choice("param_float", self.ranges["param_float"])

          # if param_categorical==A, then param_float is active
          param_categorical.add_condition('A', [param_float])

          example = Combination([param_int, param_categorical])
          exampleroot = Choice("function", ["classify_code"])
          exampleroot.add_condition("classify_code", [example])

          self.root = exampleroot

Since sklearn uses consistent framework across classifiers (fit, predict, etc.), atm creates a pipeline with the classifier and uses the sklearn standard framework to learn the classifier and make predictions.



Adding a Parameter Selector
---------------------------
A parameter selector can be created by creating a class which inherits the ``SamplesSelector`` class.
The class must have a ``select`` function which returns the chose parameters.
A few examples which use a Gaussian Process to choose parameters is shown below.

.. literalinclude:: ../../atm/selection/samples/gp_ei.py


Changing the Acquisition Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Gaussian Process Expected Improvement selection scheme makes use of an acquisition function to decide which parameter set will offer the best performance improvement.
The current acquisition function (seen below) makes use of the predicted performance and the confidence of the performance to create a new metric for deciding which parameter set will likely offer the best performance.
This metric can be altered depending on the needs of a particular problem in the ``acquisition.py`` file in the ``atm/selection`` folder.

.. literalinclude:: ../../atm/selection/acquisition.py



Adding a Frozens (Hyperpartition) Selector
------------------------------------------
A parameter selector can be created by creating a class which inherits the ``FrozenSelector`` class.
The class must have a ``select`` function which returns the chose parameters.
An example which uses the UCB1 algorithm to choose the hyperpartition is shown below.

.. literalinclude:: ../../atm/selection/frozens/ucb1.py

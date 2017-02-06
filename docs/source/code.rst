delphi
======

Adding a classifier
-------------------


Decide name for classifier (e.g., KNeighborsClassifier) and emuerator (e.g., EnumeratorKNN).
Enter these names in the `LEARNER_CODE_CLASS_MAP` and `ENUMERATOR_CODE_CLASS_MAP` variables in `delphi/mapping.py`

Create Enumerator for class in the `delphi/enumeration/classification` folder.
In this file:

a) define the ranges of the parameters in a variable called `DEFAULT_RANGES`
b) define the parameter types (int, float, etc.) in a variable called `DEFAULT_KEYS`
c) define the Conditional Parameter Tree (CPT) in a function called `create_cpt`

.. code:: python

  from delphi.cpt import Choice, Combination
  from delphi.enumeration import Enumerator
  from delphi.enumeration.classification import ClassifierEnumerator
  from delphi.key import Key, KeyStruct
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
          "param_float" : KeyStruct(DEFAULT_RANGES["param_float"], Key.TYPE_STRING, False),
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

          example = Combination([param_int, param_categorical, param_float])
          exampleroot = Choice("function", ["classify_example"])
          exampleroot.add_condition("classify_knn", [example])

          self.root = exampleroot

Currently, the delphi system uses classifiers in the `sklearn` package, so only the name of the classifier is needed.
Since sklearn uses consistent framework across classifiers (fit, predict, etc.), delphi creates an object using the name of classifier and uses the sklearn standard framework to learn the classifier and make predictions.

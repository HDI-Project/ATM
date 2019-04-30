Adding a BTB Selector or Tuner
==============================

BTB is the metamodeling library and framework at the core of ATM. It defines two
general abstractions:

1. A *selector* chooses one of a discrete set of possibilities based on
   historical performance data for each choice. ATM uses a selector before
   training each classifier to choose which hyperpartition to try next.

2. A *tuner* generates a metamodel which tries to predict the score that a set
   of numeric hyperparameters will achieve, and can generate a set of
   hyperparameters which are likely to do well based on that model. After ATM
   has chosen a hyperpartition, it uses a tuner to choose a new set of
   hyperparameters within the hyperpartition's scope.

Like with `methods <add_method.html>`_, ATM allows domain experts and tinkerers
to build their own selectors and tuners. At a high level, you just need to
define a subclass of ``btb.Selector`` or ``btb.Tuner`` in a new python file and
create a new datarun with the 'selector' or 'tuner' set to
"path/to/your_file.py:YourClassName".

*More to come... stay tuned!*

.. Creating a hyperpartition Selector
   ----------------------------------
   A parameter selector can be created by creating a class which inherits the ``btb.Selector`` class. The class must have a ``select`` method which returns the chose parameters.


.. Changing the acquisition function
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   The Gaussian Process Expected Improvement selection scheme makes use of an acquisition function to decide which parameter set will offer the best performance improvement.  The current acquisition function (seen below) uses the predicted performance and the confidence in that prediction to decide which hyperpartition to try next. This metric can be altered depending on the needs of a particular problem by modifying.


.. Creating a hyperparameter Tuner
   -------------------------------
   A parameter selector can be created by creating a class which inherits the ``btb.Tuner`` class. The class must have a ``select()`` method which returns the chose parameters.  An example which uses the UCB1 algorithm to choose the hyperpartition is shown below.

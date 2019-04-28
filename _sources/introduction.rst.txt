ATM: Scalable model selection and tuning
========================================

Auto Tune Models (ATM) is an AutoML system designed with ease of use in mind. In
short, you give ATM a classification problem and a dataset as a CSV file, and
ATM will try to build the best model it can. ATM is based on a `paper
<https://cyphe.rs/static/atm.pdf>`_ of the same name, and the project is part of
the `Human-Data Interaction (HDI) Project <https://dai.lids.mit.edu/>`_ at MIT.

To download ATM and get started quickly, head over to the `setup <setup.html>`_ section.

Background
----------
`AutoML <http://www.ml4aad.org/automl/>`_ systems attempt to automate part or all
of the machine learning pipeline, from data cleaning to feature extraction to
model selection and tuning. ATM focuses on the last part of the machine-learning
pipeline: model selection and hyperparameter tuning.

Machine learning algorithms typically have a number of parameters (called
*hyperparameters*) that must be chosen in order to define their behavior. ATM
performs an intelligent search over the space of classification algorithms and
hyperparameters in order to find the best model for a given prediction problem.
Essentially, you provide a dataset with features and labels, and ATM does the
rest.

Our goal: flexibility and power
-------------------------------

Nearly every part of ATM is configurable. For example, you can specify which
machine-learning algorithms ATM should try, which metrics it computes (such as
F1 score and ROC/AUC), and which method it uses to search through the space of
hyperparameters (using another HDI Project library, `BTB
<https://github.com/HDI-Project/btb>`_). You can also constrain ATM to find the
best model within a limited amount of time or by training a limited amount of
total models.

ATM can be used locally or on a cloud-computing cluster with AWS.
Currently, ATM only works with classification problems, but the project is under
active development. If you like the project and would like to help out, check
out our guide to `contributing <contributing.html>`_!

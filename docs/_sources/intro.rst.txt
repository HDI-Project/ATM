Introduction
============

ATM is an AutoML system designed with ease of use in mind.

[AutoML](http://www.ml4aad.org/automl/) systems attempt to automate part or all
of the machine learning pipeline, from data cleaning to feature extraction to
model selection and tuning. ATM focuses on the last part of the machine-learning
pipeline: model selection and hyperparameter tuning. 

Machine learning algorithms typically have a number of parameters (called
*hyperparameters*) that must be selected in order to define their behavior. ATM
performs an intelligent search over the space of classification algorithms and
hyperparameters in order to find the best model for a given prediction problem.
Essentially, you provide a dataset with features and labels, and ATM does the
rest.

You can tune the way ATM works, for example, by telling it which
machine-learning methods to try, or by setting the method by which ATM searches
through the hyperparameter space (using another library,
[BTB](https://github.com/HDI-Project/btb)). You can also constrain ATM to find
the best model within a limited amount of time or by training a limited amount
of total models.

ATM can be used locally or on a cloud-computing cluster with AWS. 
Currently, ATM only works with classification problems, but the project is under
active development. Feel free to contribute by opening an issue or submitting a
pull request. 

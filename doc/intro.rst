Introduction
============

Delphi is an AutoML system designed with ease of use in mind.
Typically a Machine Learning algorithm has parameters that must be tuned.
Delphi, like many other AutoML systems, abstracts this tuning sequence away from the end user.
Delphi takes in data with pre-extracted feature vectors and labels in a simple CSV file format.
In the end, Delphi returns the best classifier with parameters for a particular input feature set.
There are still some parameters, but they are of a more intuitive nature (*e.g.*, maximum number of classifier runs, metric of performance, *etc.*).
It can operate in two modes: Local and Cloud.

Local Mode
----------
In Local Mode, all processing is done on a single machine.
Depending on computer capabilities, this option may take longer.

Cloud Mode
----------
In Cloud Mode, Delphi uploads the data to the cloud and does all processing in the cloud.
Delphi is currently setup to run with Amazon Web Services (AWS), but can be easily extended to other architectures.
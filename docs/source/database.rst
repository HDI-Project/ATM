Guide to the ModelHub database
==============================

Datasets
--------
Fields:

- dataset_id
- name
- train_path
- test_path
- class_column


Dataruns
--------
datarun_id
dataset_id
hyperpartition_selection_scheme
t_s
hyperparameters_tuning_scheme
r_minimum
priority
start_time
end_time
budget_type
budget_amount


Hyperpartitions
---------------
hyperparition_id
datarun_id
method
partition_hyperparameter_values
conditional_hyperparameters


Classifiers
-----------

classifier_id
datarun_id
hyperpartition_id
model_location
metrics_location
cv_judgment_metric
test_judgment_metric
hyperparameters_values
start_time
end_time
status
error_message

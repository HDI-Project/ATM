close all; clear; clc;

data = readtable('datasetinfo.csv');

data = data(data.MissingValues == 0,:);
data = data((data.Classes > 1 & data.Classes < 5),:);

data = sortrows(data, 'Instances');

writetable(data, 'datasets.csv');
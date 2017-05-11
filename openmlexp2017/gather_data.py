import openml
import operator
import csv

def get_rank_dict(d, descending = False, rank_as_key = False):
	new_dict = {}
	sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse = descending)
	for i in xrange(len(sorted_d)):
		if rank_as_key:
			new_dict[i] = sorted_d[i][0]
		else:
			new_dict[sorted_d[i][0]] = i

	return new_dict

def select_top_n(d, n):
	top_list = []
	for i in xrange(n):
		top_list.append(d[i])
	return top_list

def write_csv(name, id_list, datasets):
	ofile = open(name + '.csv', 'wb')
	writer = csv.writer(ofile, delimiter=",")

	writer.writerow(['Dataset ID', 'Dataset Name','Number of Instances', 'Number of Instances With Missing Values', 'Number of Classes', 'Number of Features'])

	for did in id_list:
		dataset = datasets[did]
		writer.writerow([did, dataset['name'], dataset['NumberOfInstances'], dataset['NumberOfInstancesWithMissingValues'], dataset['NumberOfClasses'], dataset['NumberOfFeatures']])

	ofile.close()

apikey = 'c0bbf61f0ca7139a3db5562edcbe10e5'
openml.config.apikey = apikey

datasets = openml.datasets.list_datasets()
metric_dict = {}

for key in datasets:
	dataset = datasets[key]

	try:
		if dataset['status'] != 'active':
			continue
		else:
			data_id = key
			num_instances = dataset['NumberOfInstances']
			num_missing_instances = dataset['NumberOfInstancesWithMissingValues']
			num_features = dataset['NumberOfFeatures']

			metric_dict[data_id] = num_instances / (num_missing_instances + 1.0)

	except KeyError as e:
		print "missing " + str(e)

metric_dict_ranked = get_rank_dict(metric_dict, True, True)

top_1000 = select_top_n(metric_dict_ranked, 1000)

write_csv('datasets', top_1000, datasets)






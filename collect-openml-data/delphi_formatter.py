import openml
import csv

apikey = 'c0bbf61f0ca7139a3db5562edcbe10e5'
openml.config.apikey = apikey

file = open('chosen_ids.txt', 'r')
item_list = file.read().split()
name_list = {}

datasets = openml.datasets.list_datasets()

for item in item_list:
	print item

	did = int(item)

	try:
		dataset = openml.datasets.get_dataset(did).get_data(return_attribute_names=True)
		dataset_name = datasets[did]['name']
		if dataset_name in name_list:
			name_list[dataset_name] += 1
		else:
			name_list[dataset_name] = 1
		name = dataset_name + '_' + str(name_list[dataset_name]) + '.csv'
		header = dataset[1]
		header[-1] = 'class'

		ofile = open(name, 'wb')
		writer = csv.writer(ofile, delimiter=",")

		writer.writerow(header)
		for row in dataset[0]:
			writer.writerow(row)

		ofile.close()

	except Exception as e:
		print 'Exception:', e

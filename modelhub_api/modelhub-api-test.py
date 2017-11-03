import modelhub_api.queries as mh

print 'These are the functions available in ATM:'

funcs = mh.get_functions()
print '\t|{}|'.format('-'*47)
print '\t| {0: >15} | {1: <27} |'.format('function_id', 'name')
print '\t|{}|'.format('-'*47)
for func in funcs:
    print '\t| {0: >15} | {1: <27} |'.format(func[0], func[1])
print '\t|{}|'.format('-' * 47)



print '\nThese are 5 datasets in the ModelHub for which SVM & KNN have been applied:'

datasets = mh.get_datasets_info(n=5, function_ids= ['classify_svm', 'classify_knn'])
print '\t|{}|'.format('-'*34)
print '\t| {0: >2} | {1: <27} |'.format('ID', 'Dataset')
print '\t|{}|'.format('-'*34)
for dataset in datasets:
    print '\t| {0: >2} | {1: <27} |'.format(dataset[0],dataset[1])
print '\t|{}|'.format('-' * 34)



print '\nHere are some SVM & KNN results for dataset 3:'

classifier_ids = mh.get_classifier_ids(n = 5, dataset_ids = 3, function_ids= ['classify_svm', 'classify_knn'])
classifier_structs = mh.get_classifier_details(classifier_ids)

print '\t|{}|'.format('-'*29)
print '\t| {0: >15} | {1: <9} |'.format('Function', 'Test Acc.')
print '\t|{}|'.format('-'*29)
for classifier_struct in classifier_structs:
    print '\t| {0: >15} | {1:9.2f} |'.format(classifier_struct.function_id,classifier_struct.test_accuracy)
print '\t|{}|'.format('-' * 29)

print '\nClassifier Details:\n'
for classifier_struct in classifier_structs:
    print classifier_struct
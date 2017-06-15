import modelhub_api.queries as mh

print 'These are the algorithms available in Delphi:'

algs = mh.get_algorithms()
print '\t|{}|'.format('-'*47)
print '\t| {0: >15} | {1: <27} |'.format('code', 'name')
print '\t|{}|'.format('-'*47)
for alg in algs:
    print '\t| {0: >15} | {1: <27} |'.format(alg[0],alg[1])
print '\t|{}|'.format('-' * 47)



print '\nThese are 5 datasets in the ModelHub for which SVM & KNN have been applied:'

datasets = mh.get_datasets(n=5, codes = ['classify_svm', 'classify_knn'])
print '\t|{}|'.format('-'*34)
print '\t| {0: >2} | {1: <27} |'.format('ID', 'Dataset')
print '\t|{}|'.format('-'*34)
for dataset in datasets:
    print '\t| {0: >2} | {1: <27} |'.format(dataset[0],dataset[1])
print '\t|{}|'.format('-' * 34)



print '\nHere are some SVM & KNN results for dataset 3:'

classifier_ids = mh.get_classifiers(n = 5, dataset_ids = 3, codes = ['classify_svm', 'classify_knn'])
classifier_structs = mh.get_classifier_details(classifier_ids)

print '\t|{}|'.format('-'*29)
print '\t| {0: >15} | {1: <9} |'.format('Algorithm', 'Test Acc.')
print '\t|{}|'.format('-'*29)
for classifier_struct in classifier_structs:
    print '\t| {0: >15} | {1:9.2f} |'.format(classifier_struct.algorithm_code,classifier_struct.test_accuracy)
print '\t|{}|'.format('-' * 29)
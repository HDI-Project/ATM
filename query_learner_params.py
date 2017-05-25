from delphi.database import *


learner_id_number = 31

learner = GetLearner(learner_id_number)

print 'The param64 values are:'
for key, value in learner.params.iteritems():
    print '\t{} = {}'.format(key, value)

print 'The trainable_param64 values are:'
for key, value in learner.trainable_params.iteritems():
    print '\t{} = {}'.format(key, value)

if learner.errored is None:
    print 'Test Accuracy was {}'.format(learner.test)
else:
    print 'Note: Learner {} errored and record a performance'.format(learner_id_number)
from delphi.database import *
from delphi.utilities import Base64ToObject


learner_id_number = 31

learner = GetLearner(learner_id_number)
frozen = GetFrozenSet(learner.frozen_set_id,False)


print 'The param64 values are:'
for key, value in learner.params.iteritems():
    print '\t{} = {}'.format(key, value)

print 'The trainable_param64 values are:'
for key, value in learner.trainable_params.iteritems():
    print '\t{} = {}'.format(key, value)

print 'The frozens64 values are:'
for idx in range(len(frozen.frozens)):
    print '\t{} = {}'.format(frozen.frozens[idx][0], frozen.frozens[idx][1])

if learner.errored is None:
    print 'Test Accuracy was {}'.format(learner.test)
else:
    print 'Note: Learner {} errored and record a performance'.format(learner_id_number)
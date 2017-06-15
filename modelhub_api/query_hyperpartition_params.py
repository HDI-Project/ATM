from delphi.database import *


frozen_set_ids = range(1,103)

print 'Model,Parameter Type,Parameter,Value (if categorical)/Range (if continuous)'

for id in frozen_set_ids:
    frozen = GetFrozenSet(id,False)

    first_line = '{},Categorical,'.format(frozen.algorithm)

    for idx in range(len(frozen.frozens)):
        first_line += '{},{}'.format(frozen.frozens[idx][0], frozen.frozens[idx][1])
        print first_line
        first_line = ',,'

    first_line = ',Continuous,'
    for idx in range(len(frozen.optimizables)):
        keystruct_obj = frozen.optimizables[idx][1]
        range_start = keystruct_obj.range[0]
        range_end = keystruct_obj.range[1]
        first_line += '{},{}-{}'.format(frozen.optimizables[idx][0], range_start, range_end)
        print first_line
        first_line = ',,'
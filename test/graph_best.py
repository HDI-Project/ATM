import argparse

import matplotlib.pyplot as plt
import numpy as np

from atm.enter_data import load_config
from atm.database import Database


parser = argparse.ArgumentParser(description='''
Use matplotlib to graph the best-so-far performance of one or more dataruns.
''')
parser.add_argument('--sql-config', help='path to ModelHub SQL configuration',
                    default='config/test/sql_config.yaml')

parser.add_argument('--dataruns', type=int, nargs='+',
                    help='IDs of dataruns to graph. If None, graph all completed dataruns.')


def graph_best_so_far(db, datarun_ids=None, n_learners=100):
    lines = []
    dataruns = db.get_dataruns(ignore_pending=True, ignore_complete=False,
                               include_ids=datarun_ids)

    for r in dataruns:
        # generate a list of the "best so far" score after each learner was
        # computed (in chronological order)
        learners = db.get_learners_in_datarun(r.id)[:n_learners]
        print 'run %s: %d learners' % (r, len(learners))
        x = range(len(learners))
        y = []
        for l in learners:
            best_so_far = max(y + [l.cv_judgment_metric])
            y.append(best_so_far)

        line, = plt.plot(x, y, '-', label=str(r.id))
        lines.append(line)

    plt.xlabel('learners')
    plt.ylabel(r.metric)
    plt.legend(handles=lines)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    sql_config, _, _ = load_config(sql_path=args.sql_config)
    db = Database(**vars(sql_config))
    graph_best_so_far(db, args.dataruns)

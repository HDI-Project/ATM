import argparse
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

from atm.config import *


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

def print_summary(db, rid):
    run = db.get_datarun(rid, ignore_complete=False)
    ds = db.get_dataset(run.dataset_id)
    print
    print 'Dataset %s' % ds
    print 'Datarun %s' % run

    learners = db.get_learners_in_datarun(rid)
    print 'Learners: %d total' % len(learners)

    lid, score, std = db.get_best_so_far(run.id, run.score_target)
    print 'Best result overall: learner %d, %s = %.3f +- %.3f' % (lid,
                                                                  run.metric,
                                                                  score, std)
    dd = defaultdict(lambda: defaultdict(list))

    for l in learners:
        fs = db.get_frozen_set(l.frozen_set_id)
        dd[fs.algorithm][fs.id].append(l)

    for alg, fs_map in dd.items():
        print
        print 'algorithm %s:' % alg
        alg_running = 0
        alg_errored = 0
        alg_complete = 0
        alg_score = 0
        alg_err = 0
        alg_lid = None

        for fsid, fs_learners in fs_map.items():
            fs = db.get_frozen_set(fsid)
            running = len([l for l in fs_learners if l.status == LearnerStatus.RUNNING])
            errored = len([l for l in fs_learners if l.status == LearnerStatus.ERRORED])
            complete = len([l for l in fs_learners if l.status == LearnerStatus.COMPLETE])
            print '\tfrozen set %s:' % fs.id
            print '\t\t%d running, %d errored, %d complete' % (running, errored, complete)
            lid, score, err = db.get_best_so_far(run.id, run.score_target, fsid)
            print '\t\tBest: learner %s, %s = %.3f +- %.3f' % (lid, run.metric,
                                                               score, err)

            alg_running += running
            alg_errored += errored
            alg_complete += complete
            if alg_score - alg_err < score - err:
                alg_score = score
                alg_err = err
                alg_lid = lid

        print 'totals:'
        print '%d running, %d errored, %d complete' % (alg_running, alg_errored,
                                                       alg_complete)
        print 'Best: learner %s, %s = %.3f +- %.3f' % (alg_lid, run.metric,
                                                       alg_score, alg_err)

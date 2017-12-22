import argparse
import numpy as np

from collections import defaultdict
from multiprocessing import Process

from atm.config import *
from atm.worker import work

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def get_best_so_far(db, datarun_id):
    """
    Get a series representing best-so-far performance for datarun_id.
    """
    # generate a list of the "best so far" score after each classifier was
    # computed (in chronological order)
    classifiers = db.get_classifiers(datarun_id=datarun_id)
    print 'run %s: %d classifiers' % (datarun_id, len(classifiers))
    y = []
    for l in classifiers:
        best_so_far = max(y + [l.cv_judgment_metric])
        y.append(best_so_far)
    return y

def graph_series(length, title, **series):
    """
    Graph series of performance metrics against one another.

    length: all series will be truncated to this length
    title: what to title the graph
    **series: mapping of labels to series of performance data
    """
    if plt is None:
        raise ImportError("Unable to import matplotlib")

    lines = []
    for label, data in series.items():
        # copy up to `length` of the values in `series` into y.
        y = data[:length]
        x = range(len(y))

        # plot y against x
        line, = plt.plot(x, y, '-', label=label)
        lines.append(line)

    plt.xlabel('classifiers')
    plt.ylabel('performance')
    plt.title(title)
    plt.legend(handles=lines)
    plt.show()

def print_summary(db, rid):
    run = db.get_datarun(rid)
    ds = db.get_dataset(run.dataset_id)
    print
    print 'Dataset %s' % ds
    print 'Datarun %s' % run

    classifiers = db.get_classifiers(datarun_id=rid)
    print 'Classifiers: %d total' % len(classifiers)

    best = db.get_best_classifier(datarun_id=run.id)
    if best is not None:
        score = best.cv_judgment_metric
        err = 2 * best.cv_judgment_metric_stdev
        print 'Best result overall: classifier %d, %s = %.3f +- %.3f' %\
            (best.id, run.metric, score, err)


def print_method_summary(db, rid):
    # maps methods to sets of hyperpartitions, and hyperpartitions to lists of
    # classifiers
    alg_map = {a: defaultdict(list) for a in db.get_methods(datarun_id=rid)}

    run = db.get_datarun(rid)
    classifiers = db.get_classifiers(datarun_id=rid)
    for l in classifiers:
        hp = db.get_hyperpartition(l.hyperpartition_id)
        alg_map[hp.method][hp.id].append(l)

    for alg, hp_map in alg_map.items():
        print
        print 'method %s:' % alg

        classifiers = sum(hp_map.values(), [])
        errored = len([l for l in classifiers if l.status ==
                       ClassifierStatus.ERRORED])
        complete = len([l for l in classifiers if l.status ==
                        ClassifierStatus.COMPLETE])
        print '\t%d errored, %d complete' % (errored, complete)

        best = db.get_best_classifier(datarun_id=rid, method=alg)
        if best is not None:
            score = best.cv_judgment_metric
            err = 2 * best.cv_judgment_metric_stdev
            print '\tBest: classifier %s, %s = %.3f +- %.3f' % (best, run.metric,
                                                                score, err)

def print_hp_summary(db, rid):
    run = db.get_datarun(rid)
    classifiers = db.get_classifiers(datarun_id=rid)

    part_map = defaultdict(list)
    for c in classifiers:
        hp = c.hyperpartition_id
        part_map[hp].append(c)

    for hp, classifiers in part_map.items():
        print
        print 'hyperpartition', hp
        print db.get_hyperpartition(hp)

        errored = len([c for c in classifiers if c.status ==
                       ClassifierStatus.ERRORED])
        complete = len([c for c in classifiers if c.status ==
                        ClassifierStatus.COMPLETE])
        print '\t%d errored, %d complete' % (errored, complete)

        best = db.get_best_classifier(datarun_id=rid, hyperpartition_id=hp)
        if best is not None:
            score = best.cv_judgment_metric
            err = 2 * best.cv_judgment_metric_stdev
            print '\tBest: classifier %s, %s = %.3f +- %.3f' % (best, run.metric,
                                                                score, err)

def work_parallel(db, datarun_ids=None, aws_config=None, n_procs=4):
    print 'starting workers...'
    kwargs = dict(db=db, datarun_ids=datarun_ids, save_files=False,
                  choose_randomly=True, cloud_mode=False,
                  aws_config=aws_config, wait=False)

    if n_procs > 1:
        # spawn a set of worker processes to work on the dataruns
        procs = []
        for i in range(n_procs):
            p = Process(target=work, kwargs=kwargs)
            p.start()
            procs.append(p)

        # wait for them to finish
        for p in procs:
            p.join()
    else:
        work(**kwargs)

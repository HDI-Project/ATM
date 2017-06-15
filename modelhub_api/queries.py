from delphi.database import *
from sqlalchemy import and_

class ClassifierInfo:
    def __init__(self):
        self.classifier_id = -1
        self.dataset_id = -1
        self.algorithm_code = ''
        self.parameters = []
        self.train_accuracy = -1.0
        self.train_std = -1.0
        self.test_accuracy = -1.0


def get_algorithms():
    session = None
    algorithms = []
    try:
        session = GetConnection()
        algorithms = session.query(Algorithm).all()
    except:
        print "Error in get_algorithms:" % traceback.format_exc()
    finally:
        if session:
            session.close()

    if not algorithms:
        return []

    algorithm_tuple_list = []
    for algorithm in algorithms:
        algorithm_tuple_list.append((algorithm.code, algorithm.name))

    return algorithm_tuple_list

def get_datasets(n=None, codes=None):
    session = None
    datasets = []
    try:
        session = GetConnection()

        datarun_query = session.query(Datarun)
        frozens_query = session.query(FrozenSet.datarun_id)

        # no arguments
        if(n == None and codes == None):
            datasets = datarun_query.all()
        # only n given (i.e. only codes is None)
        elif(codes==None):
            datasets = datarun_query.limit(n).all()
        # if codes are given (for n given or not)
        else:
            dataset_ids = set()
            for code in codes:
                temp_id_list = frozens_query.filter(FrozenSet.algorithm == code).distinct().all()
                for id in temp_id_list:
                    if id[0] not in dataset_ids:
                        dataset_ids.add(id[0])
            # if n not given
            if(n == None):
                datasets = datarun_query.filter(Datarun.id.in_(list(dataset_ids))).all()
            # if n given
            else:
                datasets = datarun_query.filter(Datarun.id.in_(list(dataset_ids))).limit(n).all()

        session.close()

    except Exception:
        print "Error in GetDatarun():", traceback.format_exc()

    finally:
        if session:
            session.close()

    if not datasets:
        return []

    datasets_tuple_list = []
    for dataset in datasets:
        datasets_tuple_list.append((dataset.id, dataset.name))
    return datasets_tuple_list

def get_classifiers(n=None, dataset_ids=None, codes=None):
    session = None
    learners = []

    if type(dataset_ids) is int:
        dataset_ids = [dataset_ids]

    try:
        session = GetConnection()

        if n==None and dataset_ids==None and codes==None:
            learners = session.query(Learner).filter(Learner.is_error == 0).all()
        elif dataset_ids==None and codes==None:
            learners = session.query(Learner).filter(Learner.is_error == 0).limit(n).all()
        elif n==None and codes==None:
            learners = session.query(Learner).filter(and_(Learner.datarun_id.in_(dataset_ids), Learner.is_error == 0)).all()
        elif codes==None:
            learners = session.query(Learner).filter(and_(Learner.datarun_id.in_(dataset_ids), Learner.is_error == 0)).limit(n).all()
        elif n==None and dataset_ids==None:
            learners = session.query(Learner).filter(and_(Learner.algorithm.in_(codes), Learner.is_error == 0)).all()
        elif dataset_ids==None:
                learners = session.query(Learner).filter(and_(Learner.algorithm.in_(codes), Learner.is_error == 0)).limit(n).all()
        elif n==None:
            learners = session.query(Learner).filter(and_(Learner.algorithm.in_(codes), Learner.is_error == 0, Learner.datarun_id.in_(dataset_ids))).all()
        else:
            learners = session.query(Learner).filter(and_(Learner.algorithm.in_(codes), Learner.is_error == 0, Learner.datarun_id.in_(dataset_ids))).limit(n).all()

    except:
        print "Error in get_classifiers:" % traceback.format_exc()
    finally:
        if session:
            session.close()

    classifier_id_list = []
    for learner in learners:
        if learner.is_error == 0:
            classifier_id_list.append(int(learner.id))

    return classifier_id_list

def get_classifier_details(classifier_ids=None):
    classifier_structs = []

    if(type(classifier_ids) == list):
        for classifier_id in classifier_ids:
            classifier_structs.append(get_classifier_struct(classifier_id))

    if(type(classifier_ids) == int):
        classifier_structs = get_classifier_struct(classifier_ids)

    return classifier_structs

def get_classifier_struct(classifier_id):
    learner = GetLearner(classifier_id)

    if learner.is_error:
        return None

    frozen = GetFrozenSet(classifier_id, increment=False)

    struct = ClassifierInfo()

    struct.classifier_id = classifier_id
    struct.dataset_id = learner.datarun_id
    struct.algorithm_code = learner.algorithm
    struct.train_accuracy = learner.cv
    struct.train_std = learner.stdev
    struct.test_accuracy = learner.test

    for key, value in learner.params.iteritems():
        struct.parameters.append((key, value))

    for key, value in learner.trainable_params.iteritems():
        struct.parameters.append((key, value))

    for idx in range(len(frozen.frozens)):
        struct.parameters.append((frozen.frozens[idx][0], frozen.frozens[idx][1]))

    return struct

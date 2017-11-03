from atm.database import Database
from sqlalchemy import and_


class ClassifierInfo(object):
    def __init__(self):
        self.classifier_id = -1
        self.dataset_id = -1
        self.function_id = ''
        self.hyperparameters = []
        self.train_accuracy = -1.0
        self.train_std = -1.0
        self.test_accuracy = -1.0

    def __repr__(self):
        return 'Classifier{}'.format(self.classifier_id)

    def __str__(self):
        return self.get_string_representation()

    def get_string_representation(self):
        str = ''

        str += 'Classifier - ID = {}\n'.format(self.classifier_id)
        str += '\t{} = {}\n'.format('Dataset ID', self.dataset_id)
        str += '\t{} = {}\n'.format('Function ID', self.function_id)
        str += '\t{} = {}\n'.format('Dataset ID', self.dataset_id)
        str += '\t{} = {}\n'.format('Train Accuracy', self.train_accuracy)
        str += '\t{} = {}\n'.format('Train Standard Deviation', self.train_std)
        str += '\t{} = {}\n'.format('Test Accuracy', self.test_accuracy)
        str += '\tParameters:\n'
        for key,value in self.hyperparameters:
            str += '\t\t{} = {}\n'.format(key,value)

        return str


def get_functions():
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

    function_tuple_list = []
    for algorithm in algorithms:
        function_tuple_list.append((algorithm.code, algorithm.name))

    return function_tuple_list


def get_datasets_info(n=None, function_ids=None):
    if type(function_ids) is str:
        function_ids = [function_ids]

    session = None
    datasets = []
    try:
        session = GetConnection()

        datarun_query = session.query(Datarun)
        frozens_query = session.query(FrozenSet.datarun_id)

        # no arguments
        if (n == None and function_ids == None):
            datasets = datarun_query.all()
        # only n given (i.e. only codes is None)
        elif (function_ids == None):
            datasets = datarun_query.limit(n).all()
        # if codes are given (for n given or not)
        else:
            dataset_ids = set()
            for code in function_ids:
                temp_id_list = frozens_query.filter(FrozenSet.algorithm == code).distinct().all()
                for id in temp_id_list:
                    if id[0] not in dataset_ids:
                        dataset_ids.add(id[0])
            # if n not given
            if (n == None):
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
        datasets_tuple_list.append((int(dataset.id), dataset.name))
    return datasets_tuple_list


def get_classifier_ids(n=None, dataset_ids=None, function_ids=None):
    session = None
    learners = []

    if type(dataset_ids) is int:
        dataset_ids = [dataset_ids]

    if type(function_ids) is str:
        function_ids = [function_ids]

    try:
        session = GetConnection()

        if n == None and dataset_ids == None and function_ids == None:
            learners = session.query(Learner).filter(Learner.is_error == 0).all()
        elif dataset_ids == None and function_ids == None:
            learners = session.query(Learner).filter(Learner.is_error == 0).limit(n).all()
        elif n == None and function_ids == None:
            learners = session.query(Learner).filter(
                and_(Learner.datarun_id.in_(dataset_ids), Learner.is_error == 0)).all()
        elif function_ids == None:
            learners = session.query(Learner).filter(
                and_(Learner.datarun_id.in_(dataset_ids), Learner.is_error == 0)).limit(n).all()
        elif n == None and dataset_ids == None:
            learners = session.query(Learner).filter(
                and_(Learner.algorithm.in_(function_ids), Learner.is_error == 0)).all()
        elif dataset_ids == None:
            learners = session.query(Learner).filter(
                and_(Learner.algorithm.in_(function_ids), Learner.is_error == 0)).limit(n).all()
        elif n == None:
            learners = session.query(Learner).filter(and_(Learner.algorithm.in_(function_ids), Learner.is_error == 0,
                                                          Learner.datarun_id.in_(dataset_ids))).all()
        else:
            learners = session.query(Learner).filter(and_(Learner.algorithm.in_(function_ids), Learner.is_error == 0,
                                                          Learner.datarun_id.in_(dataset_ids))).limit(n).all()

    except:
        print "Error in get_classifier_ids:" % traceback.format_exc()
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

    if (type(classifier_ids) == list):
        for classifier_id in classifier_ids:
            classifier_structs.append(get_classifier_struct(classifier_id))

    if (type(classifier_ids) == int):
        classifier_structs = get_classifier_struct(classifier_ids)

    return classifier_structs


def get_classifier_struct(classifier_id):
    learner = GetLearner(classifier_id)

    if learner.is_error:
        return None

    struct = ClassifierInfo()

    struct.classifier_id = classifier_id
    struct.dataset_id = learner.datarun_id
    struct.function_id = learner.algorithm
    struct.train_accuracy = learner.cv
    struct.train_std = learner.stdev
    struct.test_accuracy = learner.test

    for key, value in learner.params.iteritems():
        struct.hyperparameters.append((key, value))

    # for key, value in learner.trainable_params.iteritems():
    #     struct.parameters.append((key, value))
    #
    # for idx in range(len(frozen.frozens)):
    #     struct.parameters.append((frozen.frozens[idx][0], frozen.frozens[idx][1]))

    return struct

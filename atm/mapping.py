from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from nolearn.dbn import DBN

from atm.enumeration import Enumerator
from atm.enumeration.classification import ClassifierEnumerator
from atm.enumeration.classification.svm import EnumeratorSVC
from atm.enumeration.classification.tree import EnumeratorDTC, EnumeratorRFC,\
                                                EnumeratorETC
from atm.enumeration.classification.probabilistic import EnumeratorGNB,\
                                                         EnumeratorBNB,\
                                                         EnumeratorMNB
from atm.enumeration.classification.logistic import EnumeratorLRC
from atm.enumeration.classification.nn import EnumeratorDBN, EnumeratorMLP
from atm.enumeration.classification.knn import EnumeratorKNN
from atm.enumeration.classification.gd import EnumeratorSGDC, EnumeratorPAC
from atm.enumeration.classification.gp import EnumeratorGPC
from atm.wrapper import Wrapper

# sample tuning
from btb.tuning.constants import Tuners
from btb.tuning import Uniform as UniformTuner, Grid, GP, GPEi, GPEiVelocity
# frozen selectors
from btb.selection.constants import Selectors
from btb.selection import Uniform as UniformSelector, UCB1,\
                                     BestKReward, BestKVelocity,\
                                     RecentKReward, RecentKVelocity,\
                                     HierarchicalByAlgorithm, PureBestKVelocity

class Mapping:
    LEARNER_CODE_CLASS_MAP = {
        ClassifierEnumerator.SVC: SVC,
        ClassifierEnumerator.KNN: KNeighborsClassifier,
        ClassifierEnumerator.GAUSSIAN_NAIVE_BAYES: GaussianNB,
        ClassifierEnumerator.MULTINOMIAL_NAIVE_BAYES: MultinomialNB,
        ClassifierEnumerator.BERNOULLI_NAIVE_BAYES: BernoulliNB,
        ClassifierEnumerator.SGD: SGDClassifier,
        ClassifierEnumerator.DECISION_TREE: DecisionTreeClassifier,
        ClassifierEnumerator.RANDOM_FOREST: RandomForestClassifier,
        ClassifierEnumerator.DBN: DBN,
        ClassifierEnumerator.MLP: MLPClassifier,
        ClassifierEnumerator.GPC: GaussianProcessClassifier,
        ClassifierEnumerator.EXTRA_TREES: ExtraTreesClassifier,
        ClassifierEnumerator.PASSIVE_AGGRESSIVE: PassiveAggressiveClassifier,
        ClassifierEnumerator.LRC: LogisticRegression}

    ENUMERATOR_CODE_CLASS_MAP = {
        ClassifierEnumerator.SVC: EnumeratorSVC,
        ClassifierEnumerator.DECISION_TREE: EnumeratorDTC,
        ClassifierEnumerator.KNN: EnumeratorKNN,
        ClassifierEnumerator.GAUSSIAN_NAIVE_BAYES: EnumeratorGNB,
        ClassifierEnumerator.MULTINOMIAL_NAIVE_BAYES: EnumeratorMNB,
        ClassifierEnumerator.BERNOULLI_NAIVE_BAYES: EnumeratorBNB,
        ClassifierEnumerator.SGD: EnumeratorSGDC,
        ClassifierEnumerator.RANDOM_FOREST: EnumeratorRFC,
        ClassifierEnumerator.DBN: EnumeratorDBN,
        ClassifierEnumerator.MLP: EnumeratorMLP,
        ClassifierEnumerator.GPC: EnumeratorGPC,
        ClassifierEnumerator.EXTRA_TREES: EnumeratorETC,
        ClassifierEnumerator.PASSIVE_AGGRESSIVE: EnumeratorPAC,
        ClassifierEnumerator.LRC: EnumeratorLRC}

    TUNERS_MAP = {
        Tuners.UNIFORM: UniformTuner,
        Tuners.GP: GP,
        Tuners.GP_EI: GPEi,
        Tuners.GP_EI_VEL: GPEiVelocity,
        Tuners.GRID: UniformTuner,
    }

    SELECTORS_MAP = {
        Selectors.UNIFORM: UniformSelector,
        Selectors.UCB1: UCB1,
        Selectors.BEST_K: BestKReward,
        Selectors.BEST_K_VEL: BestKVelocity,
        Selectors.PURE_BEST_K_VEL: PureBestKVelocity,
        Selectors.RECENT_K: RecentKReward,
        Selectors.RECENT_K_VEL: RecentKVelocity,
        Selectors.HIER_ALG: HierarchicalByAlgorithm,
        #Selectors.HIER_RAND: HierarchicalRandom,
    }


def create_wrapper(params, judgment_metric):
    learner_class = Mapping.LEARNER_CODE_CLASS_MAP[params["function"]]
    return Wrapper(params["function"], judgment_metric, params, learner_class)


def frozen_sets_from_algorithm_codes(codes, verbose=False):
    """
    Takes in string codes and outputs frozen sets.
    """
    algorithms = []
    for code in codes:
        enumerator = Mapping.ENUMERATOR_CODE_CLASS_MAP[code]()
        algorithms.append(enumerator)

    # enumerate all frozen sets and then store them in the
    # frozen sets map which is a map:
    #
    #   algorithm_code => [
    #       categorical keys => optimizable keys, constant keys # frozen set 1
    #       etc.
    #   ]
    #
    # Each algorithm has a mapping from each unique categorical key set (the
    # frozen keys) to the corresponding continuously varying keys (the
    # optimizable keys) for which our hyperparameter optimization may take
    # place.
    frozen_sets = {}
    for algorithm in algorithms:

        # for each algorithm, get all the unique settings of
        # categorical variables as a tuple of tuples
        categoricals = algorithm.get_categorical_keys()
        optimizable = algorithm.get_optimizable_keys()
        constant_optimizables = algorithm.get_constant_optimizable_keys()

        seen_categorical_settings = set()  # set of tuple of tuples
        categoricals_to_optimizables = {}

        for params in algorithm.combinations():
            # only get the k,v pairs which are from
            # categorical keys
            categorical_set = []
            optimizable_set = []
            constant_optimizable_set = []
            for k, v in params.iteritems():
                if k in categoricals:
                    categorical_set.append((k, v))
                elif k in optimizable:
                    optimizable_set.append((k, algorithm.keys[k]))
                elif k in constant_optimizables:
                    constant_optimizable_set.append((k, algorithm.keys[k]))

            # convert to tuple of tuples so it is hashable
            categorical_set = tuple(categorical_set)
            seen_categorical_settings.add(categorical_set)
            categoricals_to_optimizables[categorical_set] = (optimizable_set, constant_optimizable_set)

        # add to our list of frozen sets
        if not algorithm.code in frozen_sets:
            frozen_sets[algorithm.code] = categoricals_to_optimizables

    if verbose:
        frozen_counter = {}
        for algorithm in algorithms:
            print "=" * 10
            print "Frozen sets for algorithm=%s\n" % algorithm.code
            if algorithm.code not in frozen_counter:
                frozen_counter[algorithm.code] = 0
            for k, v in frozen_sets[algorithm.code].iteritems():
                print "Frozen set categorical settings:", k
                print "Optimizable keys:", v[0]
                print "Constant optimizable keys:", v[1]
                print
                frozen_counter[algorithm.code] += 1

        # summary
        print "=" * 10 + " Summary " + 10 * "="
        for algorithm, fcount in frozen_counter.iteritems():
            print "Algorithm %s had %d frozen sets" % (algorithm, fcount)

    return frozen_sets

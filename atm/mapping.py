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
from btb.tuning import Uniform as UniformTuner, GP, GPEi, GPEiVelocity
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
        ClassifierEnumerator.LRC: LogisticRegression
    }



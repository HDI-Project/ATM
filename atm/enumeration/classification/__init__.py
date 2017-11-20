from atm.cpt import Choice, Combination
from atm.enumeration import Enumerator

# TODO: merge this with the definition in atm/constants
class ClassifierEnumerator(Enumerator):

    LEARNTYPE = "classification"

    SVC = "svm"
    KNN = "knn"
    GAUSSIAN_NAIVE_BAYES = "gnb"
    MULTINOMIAL_NAIVE_BAYES = "mnb"
    BERNOULLI_NAIVE_BAYES = "bnb"
    SGD = "sgd"
    DECISION_TREE = "dt"
    RANDOM_FOREST = "rf"
    DBN = "dbn"
    MLP = "mlp"
    EXTRA_TREES = "et"
    PASSIVE_AGGRESSIVE = "pa"
    LRC = "logreg"
    GPC = "gp"

    def __init__(self, hypers, categoricals):
        super(ClassifierEnumerator, self).__init__(hypers, categoricals)

    def combinations(self):
        if self.root:
            return self.root.combinations()
        return None

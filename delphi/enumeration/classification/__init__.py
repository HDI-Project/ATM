from delphi.cpt import Choice, Combination
from delphi.enumeration import Enumerator 

class ClassifierEnumerator(Enumerator):

    LEARNTYPE = "classification"
    
    SVC = "classify_svm"
    KNN = "classify_knn"
    GAUSSIAN_NAIVE_BAYES = "classify_gnb"
    MULTINOMIAL_NAIVE_BAYES = "classify_mnb"
    BERNOULLI_NAIVE_BAYES = "classify_bnb"
    SGD = "classify_sgd"
    DECISION_TREE = "classify_dt"
    RANDOM_FOREST = "classify_rf"
    DBN = "classify_dbn"
    MLP = "classify_mlp"
    EXTRA_TREES = "classify_et"
    PASSIVE_AGGRESSIVE = "classify_pa"
    LRC = "classify_logreg"
    
    def __init__(self, hypers, categoricals):
        super(ClassifierEnumerator, self).__init__(hypers, categoricals)
    
    def combinations(self):
        if self.root:
            return self.root.combinations()
        return None

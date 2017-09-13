from __future__ import division
from atm.selection.samples import *
from atm.utilities import *
from atm.database import *
from sklearn.gaussian_process import GaussianProcess
import numpy as np
import time, traceback

from atm.selection.acquisition import expected_improvement
from atm.selection.samples.uniform import UniformSampler

class GPEi(SamplesSelector):
    
    def __init__(self, **kwargs):
        """
        Needs:
        frozen_set, learners, metric, best_y
        """
        super(GPEi, self).__init__(**kwargs)
        
    def select(self):
        """
        Takes in learner objects from database that
        have been completed.
        """
        past_params = []
        learners = GetLearnersInFrozen(self.frozen_set.id)
        learners = [x for x in learners if x.completed]
        for learner in learners:
            y = float(getattr(learner, self.metric))
            past_params.append((learner.params, y))
            
        return self.do_selection(past_params)
            
        
    def do_selection(self, past_params):
        """
        Based on past parameterizations and their performances, 
        select a best candidate for evaluation by randomly generating 
        many examples and seeing which has the highest average expected
        regression value. 
        
        Example format:
        
            past_params = [
                ({...}, y1), 
                ({...}, y2), 
                ...
            ]
        """
        # etract parameters and performances
        params = [x[0] for x in past_params]
        y = np.array([x[1] for x in past_params])
        X = ParamsToVectors(params, self.frozen_set.optimizables)
        
        # train a GP
        gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget=np.finfo(np.double).eps * 1000)
        gp.fit(X, y) 
        
        # randomly generate many vectors
        candidates = GenerateRandomVectors(1000, self.frozen_set.optimizables)
        ys, stdevs = gp.predict(candidates, eval_MSE=True)
        predictions = zip(ys, stdevs)
        predictions = [expected_improvement(y, self.best_y, stdev) for (y, stdev) in predictions]
        
        #print "PREDICTIONS", np.unique(predictions)
        
        # choose one with highest average, convert, and return
        chosen = candidates[np.argmax(predictions)]
        
        return VectorBackToParams(chosen, self.frozen_set.optimizables, self.frozen_set.frozens, self.frozen_set.constants)
        
class GPEiTime(SamplesSelector):
    
    def __init__(self, **kwargs):
        """
        Needs:
        frozen_set, metric, best_y
        """
        super(GPEiTime, self).__init__(**kwargs)
        
    def select(self):
        """
        Takes in learner objects from database that
        have been completed.
        """
        past_params = []
        learners = GetLearnersInFrozen(self.frozen_set.id)
        learners = [x for x in learners if x.completed]
        for learner in learners:
            y = float(getattr(learner, self.metric))
            elapsed = (learner.completed - learner.started).total_seconds()
            if elapsed <= 0:
                elapsed = 1.0
            past_params.append((learner.params, y / elapsed))
            
        return self.do_selection(past_params)
            
        
    def do_selection(self, past_params):
        """
        Based on past parameterizations and their performances, 
        select a best candidate for evaluation by randomly generating 
        many examples and seeing which has the highest average expected
        regression value.
        
        The value to train on is EI divided by time for expected improvement
        per unit time. 
        
        Example format:
        
            past_params = [
                ({...}, y1), 
                ({...}, y2), 
                ...
            ]
        """
        # etract parameters and performances
        params = [x[0] for x in past_params]
        y = np.array([x[1] for x in past_params])
        X = ParamsToVectors(params, self.frozen_set.optimizables)
        
        # train a GP
        gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget=np.finfo(np.double).eps * 1000)
        gp.fit(X, y) 
        
        # randomly generate many vectors
        candidates = GenerateRandomVectors(1000, self.frozen_set.optimizables)
        ys, stdevs = gp.predict(candidates, eval_MSE=True)
        predictions = zip(ys, stdevs)
        predictions = [expected_improvement(y, self.best_y, stdev) for (y, stdev) in predictions]
        
        # choose one with highest average, convert, and return
        chosen = candidates[np.argmax(predictions)]
        
        return VectorBackToParams(chosen, self.frozen_set.optimizables, self.frozen_set.frozens, self.frozen_set.constants)
        
class GPEiVelocity(SamplesSelector):

    MULTIPLIER = -100
    LIMIT = 5
    
    def __init__(self, **kwargs):
        """
        Needs:
        frozen_set, learners, metric, best_y, velocity
        """
        super(GPEiVelocity, self).__init__(**kwargs)
        
    def select(self):
        """
        Takes in learner objects from database that
        have been completed.
        """

        # calculate average velocity over the k window
        session = None
        probability_of_random = 0.1
        try:
            session = Session()
            learners = session.query(Learner).\
                filter(Learner.frozen_set_id == self.frozen_set.id).\
                order_by(Learner.cv.desc()).\
                limit(GPEiVelocity.LIMIT).all()
            learners = learners[::-1]

            ################## hardcoded CV avg acc ##############!!!!!
            scores = [float(x.cv) for x in learners]
            print "GpEI scores in order increasing: %s" % scores
            vels = []
            for i in range(1, len(scores), 1):
                vels.append(scores[i] - scores[i - 1])

            print "Velocities", vels

            if vels:
                probability_of_random = np.exp(GPEiVelocity.MULTIPLIER * np.mean(vels))

        except Exception as e:
            print traceback.format_exc()
        finally:
            session.close()

        print "Probability of random: %f" % probability_of_random

        # now select a sample
        sampler = None
        if np.random.random() < probability_of_random:
            # our forward progress in this frozen set has stalled, let's 
            # introduce some randomness
            print "Choosing random so as to not stagnate!"
            sampler = UniformSampler(frozen_set=self.frozen_set)
        
        # otherwise continue with ei + time as intended
        else:
            sampler = GPEi(frozen_set=self.frozen_set,
                    metric=self.metric, best_y=self.best_y)

        return sampler.select()

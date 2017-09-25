from btb.selection.frozens import FrozenSelector, AverageVelocitiyFromList
from btb.database import *
from btb.utilities import *
import heapq
import math, random
import numpy as np

K_MIN = 3

class PureBestKVelocity(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		k, frozen_sets, metric
		"""
		super(PureBestKVelocity, self).__init__(**kwargs)
		
	def select(self):
		"""
		Keeps the frozen set counts intact but only 
		uses the top k learner's velocities over their last 
		for usage in rewards for the bandit calculation 
		"""
		print "[*] Selecting frozen with PureBestRecentKVelocity..."
		
		all_k = {} # fset id => [ floats ]
		best_k = {}  # fset id => [ k floats ]
		learners = GetLearners(self.frozen_sets[0].datarun_id)
		avg_velocities = {} # fset id => [ float ]
		
		# for each learner, add to appropriate inner list
		for learner in learners:
			if learner.frozen_set_id not in all_k:
				all_k[learner.frozen_set_id] = []
				best_k[learner.frozen_set_id] = []
				avg_velocities[learner.frozen_set_id] = []
			score = getattr(learner, self.metric)
			if IsNumeric(score):
				all_k[learner.frozen_set_id].append(float(score))
		
		# for each frozen set, heapify and retrieve the top k elements
		not_enough = False
		for fset in self.frozen_sets:
			# use heapify to get largest k elements from each all_k list
			best_k[fset.id] = heapq.nlargest(self.k, all_k.get(fset.id, []))
			avg_velocities[fset.id] = AverageVelocitiyFromList(best_k[fset.id], is_ascending=False)
			print "Frozen set %d (%s) count=%d has best k: %s => velocity: %f" % (
				fset.id, fset.algorithm, fset.trained, best_k[fset.id], avg_velocities[fset.id])
			
			if fset.trained < K_MIN:
				not_enough = True
				
		if not_enough:
			print "We don't have enough frozen trials for this k! Attempt to get all sets to same K_MIN..."
		
		# get all sets to K_MIN
		if not_enough:
			smallest_trained = 999999
			smallest_fid = -1
			random.shuffle(self.frozen_sets)
			for fset in self.frozen_sets:
				if fset.trained < smallest_trained:
					smallest_trained = fset.trained
					smallest_fid = fset.id
			return smallest_fid
		
		# purely chosen the frozen set with higest average velocity	
		else:
			max_vel = -999
			max_fid = -1
			fids = avg_velocities.keys()
			random.shuffle(fids)
			for fid in fids:
				velocity = avg_velocities[fid]
				print "is %f greater than %f?" % (velocity, max_vel)
				if velocity > max_vel:
					max_fid = fid
					max_vel = velocity
			print "Max velocity was %f for %d" % (max_vel, max_fid)
			return max_fid

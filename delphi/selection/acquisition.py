from __future__ import division
import numpy as np 
from scipy.stats import norm

standard_normal = norm()

def expected_improvement(y_est, y_best, stddev):
	"""
	Expected improvement criterion:
	http://people.seas.harvard.edu/~jsnoek/nips2013transfer.pdf

	y_est 	= what the GP estimates the value of y will be 
	y_best 	= single best y seen 
	stddev 	= uncertainty of the GP's prediction
	"""
	z_score = (y_best - y_est) / stddev
	ei = stddev * (z_score * standard_normal.cdf(z_score) + standard_normal.pdf(z_score))
	return ei
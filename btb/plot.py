import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
from sqlalchemy import asc
from btb.database import * 

def Scatter(filename, xs, ys, title, xlabel, ylabel, xlim, ylim, 
			xlog=False, ylog=False, legend_loc='lower right'):	
	"""
	xs => { description => [x values] }
	ys => { description => [corresponding y values] }
	"""

	# now plot    
	print "Plotting results..."
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	markers = ['+', '*', 'x', 'o', '>']
	colors = ['b', 'r', 'g', 'k', 'm']
	plots = []
	labels = []
	ndescriptions = len(xs.keys())
	
	for i, description in enumerate(xs.keys()):
	    marker = markers[i % len(markers)]
	    color = colors[i % len(colors)]
	    
	    if ylog:
	    	ys[description] = np.log10(ys[description])
				
		if xlog:
			xs[description] = np.log10(xs[description])
	    
	    plot = ax1.scatter(xs[description], ys[description], color=color, marker=marker, s=20)
	    plots.append(plot)
	    labels.append(description)    
	
	# set axis and title
	print "Setting axis and titles..."
	ax1.legend(plots, labels, loc='lower right')
	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(ylabel)
	ax1.set_title(title)
	
	# display as log if needed
	if xlog:
		ax1.set_xscale('log')
	if ylog:
		ax1.set_yscale('log')
		
	# set x/y limits
	ax1.set_xlim(xlim[0], xlim[1])
	ax1.set_ylim(ylim[0], ylim[1])
	        
	# save to disk
	print "Saving to disk..."
	plt.savefig(filename)
	
def Line():
	pass
class Selector(object):
	def __init__(self, **kwargs):
		for k, v in kwargs.iteritems():
			setattr(self, k, v)
	
	def select(self):
		raise NotImplementedError("Selector child class needs to implement this!")
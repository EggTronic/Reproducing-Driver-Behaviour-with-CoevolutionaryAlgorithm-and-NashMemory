# Yang Xu

# SD @ Uol
# danliangchen@gmail.com

import random
import math
import numpy as np
import theano
import theano.tensor as T
import lasagne
from model import *

class Classifier (object):	
	def __init__ (self):

		#self.input_var = T.ftensor3()
		#self.network = lasagne.layers.InputLayer((None,20,5),self.input_var)
		#self.network = lasagne.layers.RecurrentLayer(self.network, num_units=20, nonlinearity=lasagne.nonlinearities.sigmoid)

		self.input_var = T.ftensor3()
		self.network = lasagne.layers.InputLayer((None,20,5),self.input_var)
		self.network = lasagne.layers.DenseLayer(self.network, num_units=20, nonlinearity=lasagne.nonlinearities.sigmoid)
		self.network = lasagne.layers.DenseLayer(self.network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
		self.pars = lasagne.layers.get_all_params(self.network, trainable=True)
		self.fitness = 0


	def __eq__(self, other): 
		return (self.pars == other.pars)

	def __hash__(self):
		return hash(str(self.pars))

	def classify(self, model, classfy_times, time_step, noise, target_value):
		input_matrix = [model.updateState(classfy_times, time_step, noise)]
		
		# create loss function
		# target_var = T.irow('y')

		#prediction = lasagne.layers.get_output(self.network)
		#loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		#loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)

		# create parameter update expressions
		#params = lasagne.layers.get_all_params(self.network, trainable=True)
		#updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.05, momentum=0.9)

		# compile training function that updates parameters and returns training loss
		#train_fn = theano.function([self.input_var, target_var], loss, updates=updates)

		# train network 
		#loss = train_fn(input_matrix, [[target_value]])

		# use trained network for predictions
		test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		predict_fn = theano.function([self.input_var], test_prediction)	
		out_put = predict_fn(input_matrix)
		judge = out_put[0][0]

		print(judge)
		return judge

class  MixedClassifier (object):
	def __init__ (self):
		self.pars_percentage = {}

	def assign(self, classifiers):
		percentage = {}
		total = 0
		for i in range(len(classifiers)):
			self.pars_percentage[classifiers[i]] = 0
		for i in range(len(classifiers)):
			weight = random.randint(0,5)
			percentage[i] = weight
			total += weight
		if total != 0:
			for i in range(len(classifiers)):
				self.pars_percentage[classifiers[i]] = self.pars_percentage[classifiers[i]] + percentage[i]/total

	def support(self):
		s = set()
		for c in self.pars_percentage.keys():
			if self.pars_percentage[c] != 0:
				s.add(c)
		return s
		
	def not_support(self):
		s = set()
		for c in self.pars_percentage.keys():
			if self.pars_percentage[c] == 0:
				s.add(c)
		return s

	def reset(self):
		self.pars_percentage = {}


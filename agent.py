# Yang Xu
# SD @ Uol
# danliangchen@gmail.com

from model import *
from classifier import *
from pulp import *
import sys
import uuid

class Agent:
	def __init__ (self, list, name):
		self.list = list
		self.name = name
		if self.name == 'model':
			self.piN = MixedModel()
		else:
			self.piN = MixedClassifier()
		self.piN.assign(self.list)
		self.N = set()	
		self.M = set()
		self.W = None
		self.WMN = set()

	def updateWMN(self):
		self.WMN = set()
		self.N = self.piN.support()
		self.M = self.piN.not_support()
		self.WMN |= self.N
		self.WMN |= self.M 
		if self.W != None:
			self.WMN |= self.W.support()

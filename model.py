# Yang Xu
# SD @ Uol
# danliangchen@gmail.com

import random
import math
from state import *
import numpy as np

class Model (object):
	def __init__ (self):
		self.pars = [0,0,0,0,0,0]
		self.ms = [1,1,1,1,1,1]
		self.fitness = 0
		self.state = State()
		self.behaviour = []

	def __eq__(self, other): 
		return (self.pars == other.pars)

	def __hash__(self):
			return hash(str(self.pars))

	def updateState(self, classfy_times, time_step, noise):
		self.behaviour = []
		count = 0
		input_matrix = []

		# Calculate time head way
		while count < classfy_times:
			if self.state.follower.speed == 0:
				time_head_way = 6
			else:
				time_head_way = (self.state.leader.position - self.state.follower.position)/self.state.follower.speed

			# Free Driving
			if time_head_way > 3.5:
				self.behaviour.append(0)
				# print('Free')
				if (self.state.follower.speed < 40):
					self.state.follower.acceleration = self.getNormalAcceleration()
				else:
					self.state.follower.acceleration = self.getNormalDeceleration()
				

			# Car Following
			if (time_head_way <= 3.5) and (time_head_way > 0.5):
				self.behaviour.append(1)
				# print('Follow')
				# print(self.pars)
				# print(self.state.follower.speed)
				# print(self.state.distanceDiff())
				if (self.state.follower.speed >= self.state.leader.speed):
					self.state.follower.acceleration = (self.pars[0]*math.pow(self.state.follower.speed, self.pars[1])*self.state.speedDiff())/math.pow(self.state.distanceDiff(), self.pars[2])
				else:
					self.state.follower.acceleration = (self.pars[3]*math.pow(self.state.follower.speed, self.pars[4])*self.state.speedDiff())/math.pow(self.state.distanceDiff(), self.pars[5])

			# Emergency
			if (time_head_way <= 0.5):
				self.behaviour.append(2)
				# print('Emergency')
				deceleration1 = self.getNormalDeceleration()

				if (self.state.speedDiff() < 0):
					if self.state.distanceDiff() == 0:
						deceleration2 = deceleration1
					else:
						deceleration2 = self.state.leader.acceleration - 0.5*math.pow(self.state.speedDiff(), 2)/self.state.distanceDiff()
				else:
					deceleration2 = self.state.leader.acceleration + 0.25*deceleration1
				self.state.follower.acceleration = min(deceleration1, deceleration2)

			self.state.update(time_step)

			# Preparing datasets to be classified by classifiers
			input_row = []	
			if noise == True:
				input_row.append(np.float32(self.state.leader.speed*random.uniform(0.95,1.05)))
				input_row.append(np.float32(self.state.leader.position*random.uniform(0.95,1.05)))
				input_row.append(np.float32(self.state.follower.speed*random.uniform(0.95,1.05)))
				input_row.append(np.float32(self.state.follower.acceleration*random.uniform(0.95,1.05)))
				input_row.append(np.float32(self.state.follower.position*random.uniform(0.95,1.05)))
			else:
				input_row.append(np.float32(self.state.leader.speed))
				input_row.append(np.float32(self.state.leader.position))
				input_row.append(np.float32(self.state.follower.speed))
				input_row.append(np.float32(self.state.follower.acceleration))
				input_row.append(np.float32(self.state.follower.position))
			input_matrix.append(input_row)
			count += 1
			
		self.state = State()
		#print(input_matrix)
		return input_matrix

	def getNormalAcceleration(self):
		if (self.state.follower.speed < 6.096):
			return 7.8
		if (self.state.follower.speed >= 6.096) and (self.state.follower.speed <= 12.192):
			return 6.7
		if (self.state.follower.speed > 12.192):
			return 4.8

	def getNormalDeceleration(self):
		if (self.state.follower.speed < 6.1):
			return -8.7
		if (self.state.follower.speed >= 6.1) and (self.state.follower.speed < 12.2):
			return -5.2
		if (self.state.follower.speed >= 12.2) and (self.state.follower.speed < 18.3):
			return -4.4
		if (self.state.follower.speed >= 18.3) and (self.state.follower.speed < 24.4):
			return -2.9
		if (self.state.follower.speed >= 24.4):
			return -2

# Given a list of models, assign probablities for each individual
class  MixedModel (object):
	def __init__ (self):
		self.pars_percentage = {}

	def assign(self, models):
		percentage = {}
		total = 0
		for i in range(len(models)):
			self.pars_percentage[models[i]] = 0
			#print(models[i].pars)
		for i in range(len(models)):
			weight = random.randint(0,5)
			percentage[i] = weight
			total += weight
		if total != 0:
			for i in range(len(models)):
				self.pars_percentage[models[i]] = self.pars_percentage[models[i]] + percentage[i]/(2*total)
			print(self.pars_percentage)

	def support(self):
		s = set()
		for m in self.pars_percentage.keys():
			if self.pars_percentage[m] != 0:
				s.add(m)
		return s
		
	def not_support(self):
		s = set()
		for m in self.pars_percentage.keys():
			if self.pars_percentage[m] == 0:
				s.add(m)
		return s

	def reset(self):
		self.pars_percentage = {}

	def __str__(self):
		s =''
		for m in self.pars_percentage.keys():
			s += str(m.pars) + '|| '+ str(self.pars_percentage[m])+ ' |\n'

		return s

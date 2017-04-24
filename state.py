# Yang Xu
# SD @ Uol
# danliangchen@gmail.com

import math
from vehicle import *

class State (object):
	def __init__ (self):
		self.leader = Vehicle(20, 5)
		self.follower = Vehicle(0 , 8)

	def speedDiff(self):
		return (self.leader.speed - self.follower.speed)

	def distanceDiff(self):
		if (self.leader.position - self.leader.length - self.follower.position) < 0:
			return 0.1
		else:
			return (self.leader.position - self.leader.length - self.follower.position)



	def update(self, time_step):
		self.leader.position = self.leader.position + self.leader.speed*time_step + 0.5*self.leader.acceleration*math.pow(time_step, 2)
		self.leader.speed = self.leader.speed + self.leader.acceleration*time_step
		if self.leader.speed < 0:
			self.leader.speed = 0

		self.follower.position = self.follower.position + self.follower.speed*time_step + 0.5*self.follower.acceleration*math.pow(time_step, 2)
		self.follower.speed = self.follower.speed + self.follower.acceleration*time_step
		if self.follower.speed < 0:
			self.follower.speed = 0
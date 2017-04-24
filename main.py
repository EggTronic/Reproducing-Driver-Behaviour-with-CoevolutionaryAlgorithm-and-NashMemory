# Yang Xu
# SD @ Uol
# danliangchen@gmail.com

import sys
import math
import random
import operator  
from model import *
from state import *
from classifier import *
from agent import *
from itertools import *
from matplotlib.backends.backend_pdf import PdfPages
import string
import matplotlib.pyplot as plt
import numpy as np

def start(model_number, classifier_number, iteration_round, classfy_times, time_step, noise, mutate_number, car_following_model):
	def mutate_new_model(models):
		m1 = models[random.randrange(0,len(models))]
		m2 = models[random.randrange(0,len(models))]
		n1 = random.normalvariate(0, 1)

		pars2 = []
		pars3 = []
		ms2 = []
		ms3 = []

		for i in range(len(m1.pars)):
			temp = (m1.pars[i] + m2.pars[i])/2
			pars2.append(temp)
			temp = random.randrange(2)
			if temp == 0:
				ms2.append(m1.ms[i])
			else:
				ms2.append(m2.ms[i])

		r1 = 1/(2*math.sqrt(2*len(m1.pars)))
		r2 = 1/(2*math.sqrt(2*math.sqrt(len(m1.pars))))

		for i in range(len(m1.pars)):
			n2 = random.normalvariate(0, 1)
			temp = ms2[i]*math.exp(r1*n1+r2*n2)
			ms3.append(temp)
			temp = pars2[i]+ms3[i]*random.normalvariate(0, 1)
			if temp <=3 and temp >=-3:
				pars3.append(temp)
			else:
				pars3.append(pars2[i])

		m3 = Model()
		m3.pars = pars3
		m3.ms = ms3

		models.append(m3)

	def mutate_new_classifier(classifiers):
		c = Classifier()
		classifiers.append(c)

	def calculate_fitness(models,classifiers):
		i = 0
		for c in classifiers:
			i += 1
			j = 0

			result = c.classify(car_following_model,classfy_times,time_step,noise,1)
			if result >= 0.5:
				c.fitness += len(models)

			for m in models:
				j += 1
				print('------------------------|c',i,'| m',j,'|----------------------------')
				print(m.pars)
				result = c.classify(m,classfy_times,time_step,noise,0)
				if result < 0.5:
					c.fitness += 1
				else:
					m.fitness += 1

	def sort(models,classifiers):
		classifiers.sort(key=operator.attrgetter('fitness'))  
		models.sort(key=operator.attrgetter('fitness'))  		

	def payoff(mixedModel,mixedClassifier,car_following_model):
		# Generate payoff matrix
		payoff_matrix = []
		for m in mixedModel.support():
			r = []
			for c in mixedClassifier.support():
				result = c.classify(m,classfy_times,time_step,noise,0)
				if result < 0.5:
					r.append(1)
				else:
					r.append(0)
			payoff_matrix.append(r)
		print('payoff_matrix: ',payoff_matrix)

		# Calculate payoff of fake models
		payoff_fake_model = 0
		for i, m in zip(range(len(mixedModel.support())),mixedModel.support()):
			payoffRow = 0
			for j, c in zip(range(len(mixedClassifier.support())),mixedClassifier.support()):
				payoffRow += payoff_matrix[i][j]*mixedClassifier.pars_percentage[c]
			payoff_fake_model += mixedModel.pars_percentage[m]*payoffRow

		# Calculate payoff of real models
		payoff_real_model = 0
		for c in mixedClassifier.support():
			result = c.classify(car_following_model,classfy_times,time_step,noise,1)
			if result >= 0.5:
				payoff_real_model += 1*mixedClassifier.pars_percentage[c]
		payoff_real_model = payoff_real_model*0.5

		payoff = [payoff_fake_model,payoff_real_model]
		print('payoff_fake:',payoff_fake_model,' payoff_real',payoff_real_model)

		return payoff

	def searchBest1(models_top10,Matrix,classifierAgent):
		# Models try to minimize the payoff
		prob = LpProblem("solve" + str(uuid.uuid4()), LpMinimize) 
	
		# define size-many variables
		variables = []
		for i in range (len(Matrix)):
			x = LpVariable('x'+str(i), 0, 0.5)
			variables.append(x)

		v = LpVariable("v")

		# Objective 
		prob += v 

		# Constraints
		acc = 0
		for i in range(len(models_top10)):
			ac = 0
			for j, c in zip(range(len(classifierAgent.piN.support())),classifierAgent.piN.support()):
				ac += Matrix[i][j] * classifierAgent.piN.pars_percentage[c]
			acc += ac * variables[i]
		prob += v == acc

		acc = 0
		for x in variables: 
			acc += x
		prob += acc == 0.5

		GLPK().solve(prob)
		print ('------------------Best Response1 calculating -------------------------')
		response = MixedModel()
		# Solution
		for v in prob.variables():
			for i in range(len(models_top10)):
				if v.name == 'x'+ str(i) and v.varValue != 0:
					if models_top10[i] in response.pars_percentage.keys():
						response.pars_percentage[models_top10[i]] += v.varValue
					else:
						response.pars_percentage[models_top10[i]] = v.varValue

		print(response)
		return response

    
	def searchBest2(classifiers_top10,Matrix,modelAgent):
		# Classifiers try to maximize the payoff
		prob = LpProblem("solve" + str(uuid.uuid4()), LpMaximize) 
		
		# define size-many variables
		variables = []
		for i in range (len(Matrix[0])):
			y = LpVariable('y'+str(i), 0, 1)
			variables.append(y)

		v = LpVariable("v")

		# Objective 
		prob += v 

		# Constraints
		acc = 0
		for i, m in zip(range(len(modelAgent.piN.support())),modelAgent.piN.support()):
			ac = 0
			for j in range(len(classifiers_top10)):
				ac += Matrix[i][j] * variables[j]
			acc += ac * modelAgent.piN.pars_percentage[m]
		ac = 0
		for j in range(len(classifiers_top10)):
			result = classifiers_top10[j].classify(car_following_model,classfy_times,time_step,noise,1)
			if result >= 0.5:
				ac += 1*variables[j]
		acc += ac * 0.5
		prob += v == acc

		acc = 0
		for y in variables: 
			acc += y
		prob += acc == 1

		GLPK().solve(prob)
		print ('------------------Best Response2 calculating -------------------------')
		response = MixedClassifier()
		# Solution
		for v in prob.variables():
			for i in range(len(classifiers_top10)):
				if v.name == 'y'+ str(i) and v.varValue != 0:
					if classifiers_top10[i] in response.pars_percentage.keys():
						response.pars_percentage[classifiers_top10[i]] += v.varValue
					else:
						response.pars_percentage[classifiers_top10[i]] = v.varValue
		return response
	
	def nash(modelAgent,classifierAgent,models,classifiers,car_following_model):
		# Create a payoff matrix for top10 models
		matrix1 = []
		models_top10 = []
		print(len(models))
		for i in range(len(models)-10,len(models)):
			models_top10.append(models[i])
			r = []
			for c in classifierAgent.piN.support():
				result = c.classify(models[i],classfy_times,time_step,noise,0)
				if result < 0.5:
					r.append(1)
				else:
					r.append(0)
			matrix1.append(r)
		print(matrix1)

		# Create a payoff matrix for top10 classifiers
		matrix2 = []
		classifiers_top10 = []
		for i in range(len(classifiers)-10,len(classifiers)):
			classifiers_top10.append(classifiers[i])
		for m in modelAgent.piN.support():
			r = []
			for i in range(len(classifiers)-10,len(classifiers)):
				result = classifiers[i].classify(m,classfy_times,time_step,noise,0)
				if result < 0.5:
					r.append(1)
				else:
					r.append(0)
			matrix2.append(r)
		print(matrix2)

		# Get best response
		b1 = searchBest1(models_top10,matrix1,classifierAgent)
		b2 = searchBest2(classifiers_top10,matrix2,modelAgent)

		p1 = payoff(b1,classifierAgent.piN,car_following_model)
		p2 = payoff(modelAgent.piN,b2,car_following_model)


		print('---------over all payoff is ',p1,p2,' ---------')

		if  p1[0] < 0.25 and p2[0] > 0.25 and p2[1] > 0.25:
			print('----Best Response Found----')
			# Update Parallel Nash Memory Sets
			modelAgent.W = b1
			classifierAgent.W = b2
			modelAgent.updateWMN()
			classifierAgent.updateWMN()

			# Generate a payoff matrix for supports
			matrix = []
			for m in modelAgent.WMN:
				row = [] # a row
				for c in classifierAgent.WMN:
					result = c.classify(m,classfy_times,time_step,noise,0)
					if result < 0.5:
						row.append(1)
					else:
						row.append(0)
				matrix.append(row)

			row = []
			for c in classifierAgent.WMN:
				result = c.classify(car_following_model,classfy_times,time_step,noise,1)
				if result >= 0.5:
					row.append(1)
				else:
					row.append(0)
			matrix.append(row)

			# Solve model agent
			prob = LpProblem("solve" + str(uuid.uuid4()), LpMinimize)

			# Define size-many variables
			variables = []
			for i in range(len(modelAgent.WMN)):
				x = LpVariable('x'+str(i), 0, 0.5)
				variables.append(x)

			v = LpVariable("v")

			# Objective 
			prob += v 

			# Constraints
			for j in range(len(classifierAgent.WMN)):
				acc = 0
				for i in range(len(modelAgent.WMN)):
					acc += matrix[i][j] * variables[i]
				acc += matrix[len(modelAgent.WMN)][j] * 0.5
				prob += v >= acc

			acc = 0
			for x in variables: 
				acc += x
			prob += acc == 0.5

			GLPK().solve(prob)
			print ('------------------solving 1-------------------------')
			# Solution
			modelAgent.piN.reset()
			for v in prob.variables():
				for i, m in zip(range(len(modelAgent.WMN)),modelAgent.WMN):
					if v.name == 'x'+ str(i):
						modelAgent.piN.pars_percentage[m] = v.varValue

			modelAgent.updateWMN()

			# Solve Classifier Agent
			prob2 = LpProblem("solve2" + str(uuid.uuid4()), LpMaximize) # the classifier agent is always want to maximise

			# define size-many variables
			variables = []
			for i in range(len(classifierAgent.WMN)):
				x = LpVariable('y'+str(i), 0, 1)
				variables.append(x)

			v2 = LpVariable("v2")

			# Objective 
			prob2 += v2 

			# Constraints
			ac = 0
			for j in range(len(classifierAgent.WMN)):
				ac += matrix[len(modelAgent.WMN)][j] * variables[j]

			for i in range(len(modelAgent.WMN)):
				acc = 0
				for j in range(len(classifierAgent.WMN)):
					acc += matrix[i][j] * variables[j] + ac
				prob2 += v2 <= acc # the model agent will always want to minimize

			acc = 0
			for x in variables: 
				acc += x
			prob2 += acc == 1

			GLPK().solve(prob2)
			print ('------------------solving 2-------------------------')
			# Solution
			classifierAgent.piN.reset() 
			for v in prob2.variables():
				for i, c in zip(range(len(classifierAgent.WMN)),classifierAgent.WMN):
					if v.name == 'y'+ str(i):
						classifierAgent.piN.pars_percentage[c] = v.varValue

			classifierAgent.updateWMN()

			models[:] = list(modelAgent.piN.support())
			print(len(models))
			classifiers[:] = list(classifierAgent.piN.support())
			print(len(classifiers))

			for m in models:
				print(m.pars)
			for c in classifiers:
				print(c.pars)
		else:
			# No best response
			print('----No Best Response Found----')
			models[:] = list(modelAgent.piN.support())
			#print(len(models))

			classifiers[:] = list(classifierAgent.piN.support())
			#print(len(classifiers))

	def noNash(models,classifiers):
		models_top5 = []
		for i in range(len(models)-5,len(models)):
			models_top5.append(models[i])
		models[:] = models_top5
			
		classifiers_top5 = []
		for i in range(len(classifiers)-5,len(classifiers)):
			classifiers_top5.append(classifiers[i])
		classifiers[:] = classifiers_top5

	def drawPlots(classifiers,models,car_following_model,iteration,plots,l1,l2,l3,l4,l5,l6,l7,l8,l9,l,lt):
		# Pars Covergent Plot
		sum_par1 = 0
		sum_par2 = 0
		sum_par3 = 0
		sum_par4 = 0
		sum_par5 = 0
		sum_par6 = 0
		sum_num = len(models)
		for i in range(sum_num):
			sum_par1 += models[i].pars[0]
			sum_par2 += models[i].pars[1]
			sum_par3 += models[i].pars[2]
			sum_par4 += models[i].pars[3]
			sum_par5 += models[i].pars[4]
			sum_par6 += models[i].pars[5]
		avg_par1 = sum_par1/sum_num
		avg_par2 = sum_par2/sum_num
		avg_par3 = sum_par3/sum_num
		avg_par4 = sum_par4/sum_num
		avg_par5 = sum_par5/sum_num
		avg_par6 = sum_par6/sum_num

		diff1 = math.pow((avg_par1 - car_following_model.pars[0]),2)
		diff2 = math.pow((avg_par2 - car_following_model.pars[1]),2)
		diff3 = math.pow((avg_par3 - car_following_model.pars[2]),2)
		diff4 = math.pow((avg_par4 - car_following_model.pars[3]),2)
		diff5 = math.pow((avg_par5 - car_following_model.pars[4]),2)
		diff6 = math.pow((avg_par6 - car_following_model.pars[5]),2)

		variance = (diff1 + diff2 + diff3 + diff4 + diff5 + diff6)/6

		l1.append(diff1)
		l2.append(diff2)
		l3.append(diff3)
		l4.append(diff4)
		l5.append(diff5)
		l6.append(diff6)
		l.append(variance)
		lt.append(iteration)

		p1.plot(lt, l1)
		p1.plot(lt, l2)
		p1.plot(lt, l3)
		p1.plot(lt, l4)
		p1.plot(lt, l5)
		p1.plot(lt, l6)
		p1.plot(lt, l)
		p1.legend(['par1','par2','par3','par4','par5','par6','pars'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

		# Classifier Covergent Plot
		judge_sum1 = 0
		judge_sum2 = 0
		for c in classifiers:
			judge_sum1 += c.classify(car_following_model,classfy_times,time_step,noise,1)
			for m in models:
				judge_sum2 += c.classify(m,classfy_times,time_step,noise,0)

		judge_avg1 = judge_sum1/len(classifiers)
		judge_avg2 = judge_sum2/(len(classifiers)*len(models))

		l7.append(judge_avg1)
		l8.append(judge_avg2)

		p2.plot(lt,l7,color='blue')
		p2.plot(lt,l8,color='red')
		p2.legend(['real model','fake model'])

		behaviour_variacne = 0
		for m in models:
			for b1,b2 in zip(m.behaviour,car_following_model.behaviour):
				behaviour_variacne += math.pow((b1-b2),2)

		l9.append(behaviour_variacne/len(models))
		p3.plot(lt,l9,color='black')
		p3.legend(['Behaviour Variacne'])

		plots.canvas.draw()
		plt.pause(0.0001)

	# Initialize
	print('-------------- Initialize ----------------')
	count = 0
	models = []
	classifiers = []
	model_init = []
	model_init.append(Model())
	model_init.append(Model())
	#model_init.append(car_following_model)
    
    # Initialize first population of classifiers and models
	for n in range(model_number):
		models.append(Model())
		mutate_new_model(model_init)
	for n in range(classifier_number):
		classifiers.append(Classifier())

    # Initialize model agent and classifier agent
	modelAgent = Agent(model_init,'model')
	classifierAgent = Agent(classifiers,'classifier')

    # Initialize plot settings
	f, (p1, p2, p3) = plt.subplots(3, sharex=True)
	plt.xlim(1,iteration_round)
	plots = plt.gcf()
	p1.set_title('Pars_Covergency')
	p2.set_title('Classifier_Covergency')
	p3.set_title('Behaviour_Covergency')
	plots.show()
	plots.canvas.draw()

	# Initialize plot parameters
	l = []
	l1 = []
	l2 = []
	l3 = []
	l4 = []
	l5 = []
	l6 = []
	l7 = []
	l8 = []
	l9 = []
	lt = []

	# Run iteration
	while count < iteration_round:
		count += 1
		print('-------------- Iteration',count,' ----------------')
		# Coevolution Process
		for i in range(mutate_number):
			mutate_new_model(models)
			mutate_new_classifier(classifiers)
		print("----mutated----")

		calculate_fitness(models,classifiers)
		print("----fitness finished----")

		sort(models,classifiers)
		print('----co-evolve process finished-----')

		# Reset Fitness
		if count < iteration_round - 1:
			for m in models:
				m.fitness = 0
			for c in classifiers:
				c.fitness = 0

		# No Nash Memory
		# noNash(models,classifiers)

		# Nash Memory
		nash(modelAgent,classifierAgent,models,classifiers,car_following_model)
		drawPlots(classifiers,models,car_following_model,count,plots,l1,l2,l3,l4,l5,l6,l7,l8,l9,l,lt)

    # Save plot
	pp = PdfPages('5-500-1-20-False-10-20h-WithNash.pdf')
	plt.savefig(pp, format='pdf')
	pp.close()
	#print(modelAgent.piN.support())
	print('---------------- End -----------------')

def main():
	print('-------------- Start ----------------')
    # Initialize basic settings
	model_number = 5
	classifier_number = 5
	iteration_round = 500
	time_step = 1
	classfy_times = 20
	noise = False
	mutate_number = 15
	car_following_model = Model()
	car_following_model.pars = [2.15,-1.67,-0.89,1.55,1.08,1.65]
    # Run iteration
	start(model_number, classifier_number, iteration_round, classfy_times, time_step, noise, mutate_number, car_following_model)

if __name__ == "__main__":
	main()

###################################################################################
#Example to run: 
#nohup python3 -u mpeer.py DataSet/TRAIN_cholesterol.csv DataSet/TEST_cholesterol.csv prototypesProtoDash/prototypesProtodash_cholesterol 0 > Results/cholesterol &
#argv[1]: training data set
#argv[2]: test data set
#argv[3]: set of prototypes selected by ProtoDash
#argv[4]: option '0' to run the methods 30 times and calculate the medians of GFS and RMSE or option '1' to run only 1 time and plot the Pareto front graphs resulting from the MPEER methods
###################################################################################

from sklearn_extra.cluster import KMedoids
import sys
import numpy as np
from numpy import genfromtxt
from decimal import Decimal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import random
from copy import deepcopy
from scipy import stats
import math
from numpy import array
from deap import base, creator, tools, algorithms
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(suppress=True)	#print float using fixed point

#########################################################################################################
#Normalizes fidelity and stability between 0 and 1 (minmax) for gfs calculatetion
max_stability_train = -9999999999
max_stability_train_test = -9999999999
min_stability_train = 9999999999
min_stability_train_test = 9999999999

max_fidelity_train = -9999999999
max_fidelity_train_test = -9999999999
min_fidelity_train = 9999999999
min_fidelity_train_test = 9999999999

def avoid_div0(value):
	if value == 1:
		value = 0.9999999
	elif value == 0:
		value = 0.0000001
	return value
	
def normalize_fidelity_train_minMax(x):
	normalized = (x - min_fidelity_train)/(max_fidelity_train - min_fidelity_train)
	normalized = avoid_div0(normalized)
	return normalized

def normalize_fidelity_train_test_minMax(x):
	normalized = (x - min_fidelity_train_test)/(max_fidelity_train_test - min_fidelity_train_test)
	normalized = avoid_div0(normalized)
	return normalized

def normalize_stability_train_minMax(x):
	normalized = (x - min_stability_train)/(max_stability_train - min_stability_train)
	normalized = avoid_div0(normalized)
	return normalized

def normalize_stability_train_test_minMax(x):
	normalized = (x - min_stability_train_test)/(max_stability_train_test - min_stability_train_test)
	normalized = avoid_div0(normalized)
	return normalized

#########################################################################################################
def graphPlot_GFS_RMSE(pop, hof, eixox, eixoy, total_graphics, plotChart):
	x = [] #gfs
	y = [] #rmse

	for individual in pop:
		x_local,y_local = evaluation_gfs_rmse(individual)
		x.append(x_local)
		y.append(y_local)

	total_points_dominates = 0
	x_dominates = [] # gfs
	y_dominates = [] # rmse
	for individual in hof:
		x_local,y_local = evaluation_gfs_rmse(individual)
		x_dominates.append(x_local)
		y_dominates.append(y_local)

	#find the best prototype set
	best_set = -1
	gfs_best = 99999999
	num_set = 0
	x_best = 0
	y_best = 0

	for item in x_dominates:
		if item < gfs_best:
			gfs_best = item
			best_set = num_set
		num_set += 1
	x_best = x_dominates[best_set]
	y_best = y_dominates[best_set]

	prototype_best_l2 = []
	for item in hof[best_set]:
		prototype_best_l2.append(int(item))

	if plotChart == 1:
		best_set_rmse = -1
		rmse_best = 99999999
		x_best_rmse = 0
		y_best_rmse = 0
		num_set = 0

		for item in y_dominates:
			if item < rmse_best:
				rmse_best = item
				best_set_rmse = num_set
			num_set += 1
		x_best_rmse = x_dominates[best_set_rmse]
		y_best_rmse = y_dominates[best_set_rmse]

		fig = plt.figure()
		b = plt.scatter(x_dominates, y_dominates, color=['#1b9e77'], s=50)
		c = plt.scatter(x_best, y_best, color=['#d95f02'], s=80)
		d = plt.scatter(x_best_rmse, y_best_rmse, color=['#000000'], s=65)

		plt.legend((b, d, c),('Pareto Front', 'Best RMSE','Solution: Best GFS'))

		plt.xlabel(eixox)
		plt.ylabel(eixoy)

		plt.axis([0.928, 0.965, 0.445, 0.75])

		plt.savefig("graphicsNew/"+total_graphics+"_mv_"+eixox+"_2.png")
		plt.close(fig)

	return prototype_best_l2

#########################################################################################################
def graphPlot_S_F_RMSE(pop, hof, eixox, eixoy, eixoz, total_graphics, plotChart):
	x = [] #stability
	y = [] #fidelity
	z = [] #rmse

	for individual in pop:
		x_local,y_local,z_local = evaluation_e_f_rmse(individual)
		x.append(x_local)
		y.append(y_local)
		z.append(z_local)

	total_points_dominates = 0
	x_dominates = [] # stability
	y_dominates = [] # fidelity
	z_dominates = [] # rmse
	for individual in hof:
		total_points_dominates += 1
		x_local,y_local,z_local = evaluation_e_f_rmse(individual)
		x_dominates.append(x_local)
		y_dominates.append(y_local)
		z_dominates.append(z_local)

	#find the best prototype set
	best_set = -1
	gfs_best = 99999999
	num_set = 0
	x_best = 0
	y_best = 0
	z_best = 0

	for counter_test in range(total_points_dominates):
		item = ((1/math.log10(x_dominates[counter_test]))** 2 + (1/math.log10(y_dominates[counter_test]))**2) ** 0.5
		if item < gfs_best:
			gfs_best = item
			best_set = num_set
		num_set += 1
	x_best = x_dominates[best_set]
	y_best = y_dominates[best_set]
	z_best = z_dominates[best_set]

	prototype_best_l2 = []
	for item in hof[best_set]:
		prototype_best_l2.append(int(item))

	if plotChart == 1:
		best_set_rmse = -1
		rmse_best = 99999999
		x_best_rmse = 0
		y_best_rmse = 0
		z_best_rmse = 0
		num_set = 0

		for item in z_dominates:
			if item < rmse_best:
				rmse_best = item
				best_set_rmse = num_set
			num_set += 1
		x_best_rmse = x_dominates[best_set_rmse]
		y_best_rmse = y_dominates[best_set_rmse]
		z_best_rmse = z_dominates[best_set_rmse]

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		b = ax.scatter(x_dominates, y_dominates, z_dominates, color=['#1b9e77'], s=50)
		c = ax.scatter(x_best, y_best, z_best, color=['#d95f02'], s=80)
		d = ax.scatter(x_best_rmse, y_best_rmse, z_best_rmse, color=['#000000'], s=65)

		plt.legend((b, d, c),('Pareto Front', 'Best RMSE', 'Solution: Best GFS'))

		ax.set_xlabel(eixox)
		ax.set_ylabel(eixoy)
		ax.set_zlabel(eixoz)

		# rotate the axes and update
		test = 0
		for angle in range(0, 360):
			ax.view_init(30, angle)
			plt.savefig("graphicsNew/"+total_graphics+"_mv_"+eixoz+"-"+str(test)+"_2.png")
			plt.pause(.001)
			test += 1
		plt.close(fig)
		
	return prototype_best_l2

#########################################################################################################

def graphPlot_S_F(pop, hof, eixox, eixoy, total_graphics, plotChart):
	x = [] #stability
	y = [] #fidelity

	for individual in pop:
		x_local,y_local = evaluation(individual)
		x.append(x_local)
		y.append(y_local)

	total_points_dominates = 0
	x_dominates = [] # stability
	y_dominates = [] # fidelity
	rmse_dominates = [] #rmse
	for individual in hof:
		total_points_dominates += 1
		x_local,y_local = evaluation(individual)
		x_dominates.append(x_local)
		y_dominates.append(y_local)

		possibility_test = []
		for item in individual:
			possibility_test.append(int(item))
		possibility_test.sort()

		X_train_prototypes = []
		y_train_prototypes = []
		for prot in possibility_test:
			X_train_prototypes.append(X_train[prot])
			y_train_prototypes.append(y_train[prot])

		regr_prototypes_train = RandomForestRegressor(random_state=76,n_estimators = 10)
		regr_prototypes_train.fit(X_train_prototypes, y_train_prototypes)

		y_predicted_rf_train_prototypes = regr_prototypes_train.predict(X_train)
		rmse_dominates.append(np.sqrt(np.mean((y_predicted_rf_train_prototypes-y_predicted_rf_train)**2)))      #compare with what was initially predicted in training, after all I want to explain

	#find the best prototype set
	best_set = -1
	gfs_best = 99999999
	num_set = 0
	x_best = 0
	y_best = 0

	for counter_test in range(total_points_dominates):
		item = ((1/math.log10(x_dominates[counter_test]))** 2 + (1/math.log10(y_dominates[counter_test]))**2) ** 0.5
		if item < gfs_best:
			gfs_best = item
			best_set = num_set
		num_set += 1
	x_best = x_dominates[best_set]
	y_best = y_dominates[best_set]

	prototype_best_l2 = []
	for item in hof[best_set]:
		prototype_best_l2.append(int(item))


	if plotChart == 1:
		num_set = 0
		best_set_rmse = -1
		rmse_best = 99999999
		x_best_rmse = 0
		y_best_rmse = 0

		for counter_test in range(total_points_dominates):
			item = rmse_dominates[counter_test]
			if item < rmse_best:
				rmse_best = item
				best_set_rmse = num_set
			num_set += 1
		x_best_rmse = x_dominates[best_set_rmse]
		y_best_rmse = y_dominates[best_set_rmse]

		fig = plt.figure()
		b = plt.scatter(x_dominates, y_dominates, color=['#1b9e77'], s=50)
		c = plt.scatter(x_best, y_best, color=['#d95f02'], s=80)
		d = plt.scatter(x_best_rmse, y_best_rmse, color=['#000000'], s=65)

		plt.legend((b, d, c),('Pareto Front', 'Best RMSE','Solution: Best GFS'))

		plt.xlabel(eixox)
		plt.ylabel(eixoy)
		plt.axis([0.068, 0.0815, 0.0006, 0.0019])
		plt.savefig("graphicsNew/"+total_graphics+"_mv_"+eixox+"_2.png")
		plt.close(fig)

	return prototype_best_l2

#########################################################################################################
#calculatetes distances between instances
def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return float(value) ** float(root_value)

parameter_minkowski = 0.3

def calculatete_distance_minkowski_train_test(x, y):
	a = X_train[x]
	b = X_test[y]
	distance_train_test = nth_root(sum(pow(abs(u-v),parameter_minkowski) for u,v in zip(a, b)),parameter_minkowski)
	return distance_train_test

def calculatete_distance_minkowski_train_train(x, y):
	a = X_train[x]
	b = X_train[y]
	distance_train_train = nth_root(sum(pow(abs(u-v),parameter_minkowski) for u,v in zip(a, b)),parameter_minkowski)
	return distance_train_train

#########################################################################################################
#Usage on the SPEA2 Multi-Objective GA

def evaluation(individual):
	possibility_test = []
	for item in individual:
		possibility_test.append(int(item))
	possibility_test.sort()

	covered_by_train = assign_l2fs_train_ga(possibility_test)

	fidelity_median_train = calculate_fidelity_train(covered_by_train)
	stability_median_train = calculate_stability_train(covered_by_train)
	
	stability_norm_train = normalize_stability_train_minMax(stability_median_train)
	fidelity_norm_train = normalize_fidelity_train_minMax(fidelity_median_train)

	return (stability_norm_train, fidelity_norm_train)

def cxSet(ind1, ind2):
	Prob_crossover = 0.6

	child1 = []
	for item in ind1:
		child1.append(item)
	child2 = []
	for item in ind2:
		child2.append(item)

	for item in child1:
		ind1.remove(item)
	for item in child2:
		ind2.remove(item)

	for position in range(0, total_prototypes):
		if random.random() < Prob_crossover:
			if (child1[position] not in child2) and child2[position] not in child1:
				aux = child1[position]
				child1[position] = child2[position]
				child2[position] = aux
	
	for item in child1:
		ind1.add(item)
	for item in child2:
		ind2.add(item)

	return ind1, ind2

def mutSet(individual):
	Prob_mutation = 0.4
	new = []
	for item in individual:
		new.append(item)

	for item in new:
		individual.remove(item)

	for position in range(0, total_prototypes):
		if random.random() < Prob_mutation:
			new_item = random.randint(0, total_points_train-1)
			while new_item in new:
				new_item = random.randint(0, total_points_train-1)
			new[position] = new_item

	for item in new:
		individual.add(item)
	return individual,

def prototypes_MPEER_est_fid(total_graphics, plotChart):
	Prob_crossover = 0.6
	Prob_mutation = 0.4
	size_population = 300
	ngen = 75

	prototypes=[]

	creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0 ))  
	creator.create("Individual", set, fitness=creator.FitnessMulti)
	toolbox = base.Toolbox()

	# Attribute generator
	toolbox.register("attr_item", random.randrange, total_points_train)

	# Structure initializers
	toolbox.register("indices", random.sample, range(total_points_train), total_prototypes)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	toolbox.register("mate", cxSet)
	toolbox.register("mutate", mutSet)
	toolbox.register("select", tools.selSPEA2) # selSPEA2 applies to multi-objective problems
	toolbox.register("evaluate", evaluation)

	pop = toolbox.population(n= size_population)

	hof = tools.ParetoFront() # a ParetoFront may be used to retrieve the best non dominatested individuals of the evolution
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean, axis=0)
	stats.register("std", np.std, axis=0)
	stats.register("min", np.min, axis=0)
	stats.register("max", np.max, axis=0)

	MU = size_population #The number of individuals to be selected for the next generation
	LAMBDA = (size_population*2)  #The number of children to be produced in each generation
	algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, Prob_crossover, Prob_mutation, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

	print("population")
	print(pop)
	print(hof) # non-dominatested individuals' list  # the fittest value is placed on the most right side.

	prototype_best_l2 = graphPlot_S_F(pop, hof, "Stability", "Fidelity", total_graphics, plotChart)

	for item in prototype_best_l2:
		prototypes.append(int(item))

	return prototypes

#########################################################################################################
#Uso no ga multiobjetivo SPEA2 - s+f+rmse

def evaluation_e_f_rmse(individual):
	possibility_test = []
	for item in individual:
		possibility_test.append(int(item))
	possibility_test.sort()

	covered_by_train = assign_l2fs_train_ga(possibility_test)

	fidelity_median_train = calculate_fidelity_train(covered_by_train)
	stability_median_train = calculate_stability_train(covered_by_train)
	
	stability_norm_train = normalize_stability_train_minMax(stability_median_train)
	fidelity_norm_train = normalize_fidelity_train_minMax(fidelity_median_train)

	#RMSE
	X_train_prototypes = []
	y_train_prototypes = []
	for prot in possibility_test:
		X_train_prototypes.append(X_train[prot])
		y_train_prototypes.append(y_train[prot])

	regr_prototypes_train = RandomForestRegressor(random_state=76,n_estimators = 10)
	regr_prototypes_train.fit(X_train_prototypes, y_train_prototypes)

	y_predicted_rf_train_prototypes = regr_prototypes_train.predict(X_train)
	rmse_train_rf_prototypes = np.sqrt(np.mean((y_predicted_rf_train_prototypes-y_predicted_rf_train)**2))	#I compare with the initially predicted after all I want to explain the model and not the initial data

	return (stability_norm_train, fidelity_norm_train, rmse_train_rf_prototypes)


def prototypes_MPEER_e_f_rmse(total_graphics, plotChart):
	Prob_crossover = 0.6
	Prob_mutation = 0.4
	size_population = 300
	ngen = 75

	prototypes=[]

	creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0 ))  
	creator.create("Individual", set, fitness=creator.FitnessMulti)
	toolbox2 = base.Toolbox()

	# Attribute generator
	toolbox2.register("attr_item", random.randrange, total_points_train)

	# Structure initializers
	toolbox2.register("indices", random.sample, range(total_points_train), total_prototypes)
	toolbox2.register("individual", tools.initIterate, creator.Individual, toolbox2.indices)
	toolbox2.register("population", tools.initRepeat, list, toolbox2.individual)

	toolbox2.register("mate", cxSet)
	toolbox2.register("mutate", mutSet)
	toolbox2.register("select", tools.selSPEA2) # selSPEA2 applies to multi-objective problems
	toolbox2.register("evaluate", evaluation_e_f_rmse)

	pop = toolbox2.population(n= size_population)

	hof = tools.ParetoFront() # a ParetoFront may be used to retrieve the best non dominatested individuals of the evolution
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean, axis=0)
	stats.register("std", np.std, axis=0)
	stats.register("min", np.min, axis=0)
	stats.register("max", np.max, axis=0)

	MU = size_population #The number of individuals to be selected for the next generation
	LAMBDA = (size_population*2)  #The number of children to be produced in each generation
	algorithms.eaMuPlusLambda(pop, toolbox2, MU, LAMBDA, Prob_crossover, Prob_mutation, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

	print("population")
	print(pop)
	print(hof) # non-dominatested individuals' list  # the fittest value is placed on the most right side.

	prototype_best_l2 = graphPlot_S_F_RMSE(pop, hof, "Stability", "Fidelity", "RMSE", total_graphics, plotChart)

	for item in prototype_best_l2:
		prototypes.append(int(item))

	return prototypes

#########################################################################################################
#Usage in the multi-objective GA SPEA2 - gfs+rmse

def evaluation_gfs_rmse(individual):
	possibility_test = []
	for item in individual:
		possibility_test.append(int(item))
	possibility_test.sort()

	covered_by_train = assign_l2fs_train_ga(possibility_test)

	fidelity_median_train = calculate_fidelity_train(covered_by_train)
	stability_median_train = calculate_stability_train(covered_by_train)
	
	stability_norm_train = normalize_stability_train_minMax(stability_median_train)
	fidelity_norm_train = normalize_fidelity_train_minMax(fidelity_median_train)
	gfs_norm_train = ((1/math.log10(stability_norm_train))** 2 + (1/math.log10(fidelity_norm_train))**2) ** 0.5   # It must be minimized.
	
	#RMSE
	X_train_prototypes = []
	y_train_prototypes = []
	for prot in possibility_test:
		X_train_prototypes.append(X_train[prot])
		y_train_prototypes.append(y_train[prot])

	regr_prototypes_train = RandomForestRegressor(random_state=76,n_estimators = 10)
	regr_prototypes_train.fit(X_train_prototypes, y_train_prototypes)

	y_predicted_rf_train_prototypes = regr_prototypes_train.predict(X_train)
	rmse_train_rf_prototypes = np.sqrt(np.mean((y_predicted_rf_train_prototypes-y_predicted_rf_train)**2))	#I compare with the initially predicted after all I want to explain the model and not the initial data
	
	return (gfs_norm_train, rmse_train_rf_prototypes)

def prototypes_MPEER_gfs_rmse(total_graphics, plotChart):
	Prob_crossover = 0.6
	Prob_mutation = 0.4
	size_population = 300
	ngen = 75

	prototypes=[]

	creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0 ))  
	creator.create("Individual", set, fitness=creator.FitnessMulti)
	toolbox3 = base.Toolbox()

	# Attribute generator
	toolbox3.register("attr_item", random.randrange, total_points_train)

	# Structure initializers
	toolbox3.register("indices", random.sample, range(total_points_train), total_prototypes)
	toolbox3.register("individual", tools.initIterate, creator.Individual, toolbox3.indices)
	toolbox3.register("population", tools.initRepeat, list, toolbox3.individual)

	toolbox3.register("mate", cxSet)
	toolbox3.register("mutate", mutSet)
	toolbox3.register("select", tools.selSPEA2) # selSPEA2 applies to multi-objective problems
	toolbox3.register("evaluate", evaluation_gfs_rmse)

	pop = toolbox3.population(n= size_population)

	hof = tools.ParetoFront() # a ParetoFront may be used to retrieve the best non dominatested individuals of the evolution
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean, axis=0)
	stats.register("std", np.std, axis=0)
	stats.register("min", np.min, axis=0)
	stats.register("max", np.max, axis=0)

	MU = size_population #The number of individuals to be selected for the next generation
	LAMBDA = (size_population*2)  #The number of children to be produced in each generation
	algorithms.eaMuPlusLambda(pop, toolbox3, MU, LAMBDA, Prob_crossover, Prob_mutation, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

	print("population")
	print(pop)
	print(hof) # non-dominatested individuals' list  # the fittest value is placed on the most right side.

	prototype_best_l2 = graphPlot_GFS_RMSE(pop, hof, "gfs", "RMSE", total_graphics, plotChart)

	for item in prototype_best_l2:
		prototypes.append(int(item))

	return prototypes

#########################################################################################################
#use in GA

#fidelity
def calculate_fidelity_train(covered_by_train):
	fidelity_train = []
	for x in range(0, total_points_train):
		fidelity_train.append(distance_fidelity_train[covered_by_train[x]][x])
	fidelity_median_train = np.median(fidelity_train)

	return fidelity_median_train

#stability
def calculate_stability_train(covered_by_train):
	stability_train = []
	for x in range(0, total_points_train):
		stability_train.append(distance_instancias_train[int(covered_by_train[x])][x])
	stability_median_train = np.median(stability_train)

	return stability_median_train


def cal_fitness(population, size_population, fitness_population):
	fitness = []
	for possibility in range(0,size_population):
		if fitness_population[possibility] == -1:
			possibility_test = []
			for item in range(0,total_prototypes):
				possibility_test.append(int((population[possibility])[item]))

			covered_by_train = assign_l2fs_train_ga(possibility_test)

			fidelity_median_train = calculate_fidelity_train(covered_by_train)
			stability_median_train = calculate_stability_train(covered_by_train)
			
			stability_norm_train = normalize_stability_train_minMax(stability_median_train)
			fidelity_norm_train = normalize_fidelity_train_minMax(fidelity_median_train)
			gfs_norm_train = ((1/math.log10(stability_norm_train))** 2 + (1/math.log10(fidelity_norm_train))**2) ** 0.5
			
			fitness.append(gfs_norm_train)
		else:
			fitness.append(fitness_population[possibility])
	return fitness


def cal_fitness_rmse(population, size_population, fitness_population):
	fitness = []
	
	for possibility in range(0,size_population):
		if fitness_population[possibility] == -1:
			possibility_test = []
			for item in range(0,total_prototypes):
				possibility_test.append(int((population[possibility])[item]))

			seeds_prototypes = 76

			X_train_prototypes = []
			y_train_prototypes = []
			for prot in possibility_test:
				X_train_prototypes.append(X_train[prot])
				y_train_prototypes.append(y_train[prot])

			regr_prototypes_train = RandomForestRegressor(random_state=seeds_prototypes,n_estimators = 10)
			regr_prototypes_train.fit(X_train_prototypes, y_train_prototypes)

			##y predicted train
			y_predicted_rf_train_prototypes = regr_prototypes_train.predict(X_train)
			rmse_train_rf_prototypes = np.sqrt(np.mean((y_predicted_rf_train_prototypes-y_predicted_rf_train)**2))	#I compare with the initially predicted after all I want to explain the model and not the initial data

			fitness.append(rmse_train_rf_prototypes)
		else:
			fitness.append(fitness_population[possibility])
	return fitness


def crossover(pai1, pai2, Prob_crossover, total_prototypes):
	child1 = deepcopy(pai1)
	child2 = deepcopy(pai2)
	for position in range(0, total_prototypes):
		if random.random() < Prob_crossover:
			if (child1[position] not in child2) and child2[position] not in child1:
				aux = child1[position]
				child1[position] = child2[position]
				child2[position] = aux

	return child1, child2
				
def mutation(child, Prob_mutation, total_prototypes, total_points_train, size_population):
	new = deepcopy(child)
	for position in range(0, total_prototypes):
		if random.random() < Prob_mutation:
			new_item = random.randint(0, total_points_train-1)
			while new_item in new:
				new_item = random.randint(0, total_points_train-1)
			new[position] = new_item
	return new

#########################################################################################################
#choice of prototypes
def prototypes_random():
	prototypes = []
	prototypes = random.sample(range(total_points_train), int(total_prototypes))
	return prototypes

def prototypes_medoids_mink03():
	prototypes = []

	kmedoids = KMedoids(n_clusters=total_prototypes, init='random', random_state=seed, metric = 'precomputed').fit(distance_instancias_train)

	for ponto_kmedoid in kmedoids.medoid_indices_:
		prototypes.append(int(ponto_kmedoid))
	return prototypes

def prototypes_hubs(totalHubs):
	hubs = np.zeros(total_points_train)

	for p_train in range(0, total_points_train):
		aux_distances = np.zeros(total_points_train)
		for x in range(0, total_points_train):
			aux_distances[x] = distance_instancias_train[p_train][x]
		pontos_mais_proximos = np.argpartition(aux_distances, totalHubs)[:totalHubs]

		for candidato_hub in pontos_mais_proximos:
			hubs[int(candidato_hub)] += 1

	prototypes = np.argpartition(hubs, -total_prototypes)[-total_prototypes:]
	return prototypes


def prototypes_ga(tipo_fitness):
	prototypes=[]

	#create initial population
	size_population = 600
	population = np.zeros(shape=(size_population,total_prototypes))
	for aux in range(0,size_population):
		individual_random = np.zeros(total_prototypes)
		individual_random = np.random.choice(total_points_train, total_prototypes, replace=False)
		population[aux] = individual_random

	fitness_population = np.zeros(size_population)
	for aux in range(0,size_population):
		fitness_population[aux] = -1

	#evaluates the fitness of the population
	if tipo_fitness == "rmse":
		fitness_population = cal_fitness_rmse(population, size_population, fitness_population)
	else:
		fitness_population = cal_fitness(population, size_population, fitness_population)
	print ("Min initial: ",np.min(fitness_population))

	Prob_crossover = 0.8
	Prob_mutation = 0.2

	print ("Comeca a evolucao")
	generation = 0

	while min(fitness_population) > 0 and generation < 150:
		generation += 1
		print ("generation: ",generation)
		childs = np.zeros(shape=(size_population,total_prototypes))

		best_fitness = np.amin(fitness_population)
		best_individual = np.argmin(fitness_population)
		#print population
		print ("best_fitness = ",best_fitness)
		print ("best_individual = ",population[best_individual])

		total_childs = 0
		#elitism: keep the individual best
		childs[total_childs] = deepcopy(population[best_individual])
		fitness_population[total_childs] = fitness_population[best_individual]
		total_childs += 1

		while total_childs<size_population:
			#selects 2 parents among the individuals for crossover and mutation until completing size_population childs
			#tournament selection = 2
			pai1 = -1
			pai2 = -1
			size_tournament = 2

			candidatos = np.random.randint(size_population, size=size_tournament)
			fitness_candidatos = []
			for candidato in candidatos:
				fitness_candidatos.append(fitness_population[candidato])
			pai1 = candidatos[np.argmin(fitness_candidatos)]

			candidatos = np.random.randint(size_population, size=size_tournament)
			fitness_candidatos = []
			for candidato in candidatos:
				fitness_candidatos.append(fitness_population[candidato])
			pai2 = candidatos[np.argmin(fitness_candidatos)]

			if random.random() < Prob_crossover:
				child1, child2 = crossover(population[pai1], population[pai2], Prob_crossover, total_prototypes)

				childs[total_childs] = child1
				fitness_population[total_childs] = -1
				total_childs += 1	
				if total_childs<size_population:
					childs[total_childs] = child2
					fitness_population[total_childs] = -1
					total_childs += 1

		aux = 0
		for child in childs:
			if random.random() < Prob_mutation and aux != 0:	#non-zero aux conserves the individual best
				new = []
				new = mutation(child, Prob_mutation, total_prototypes, total_points_train, size_population)
				childs[aux] = deepcopy(new)
				fitness_population[aux] = -1
			aux += 1

		population = deepcopy(childs)

		if tipo_fitness == "rmse":
			fitness_population = cal_fitness_rmse(population, size_population, fitness_population)
		else:
			fitness_population = cal_fitness(population, size_population, fitness_population)
		best_fitness = np.amin(fitness_population)
		best_individual = np.argmin(fitness_population)
		print ("best_fitness = ",best_fitness)
		print ("best_individual = ", population[best_individual])

		print ("Min: ",np.min(fitness_population))
		print ("Max: ",np.max(fitness_population))
		print ("Media: ",np.mean(fitness_population))
		print ("Desvio: ",np.std(fitness_population))
		print ("median: ",np.median(fitness_population))

		prototypes = deepcopy(population[best_individual])

		print ("")

	prototypes_int = []
	for item in range(0,total_prototypes):	
		prototypes_int.append(int(prototypes[item]))
	
	return prototypes_int

#########################################################################################################
#Assignment of prototypes to explain each test point
def assign_l2fs(prototypes):
	covered_by = []
	for counter_test in range(total_points_test):
		smaller_value = 9999999999
		prototype_smaller = -1
		for item in prototypes:
			dist = l2fs_local_norm[item][counter_test]
			if dist < smaller_value:
				smaller_value = dist
				prototype_smaller = item
		covered_by.append(prototype_smaller)
	return covered_by

def assign_l2fs_train_ga(prototypes_train):
	covered_by_train = []
	for counter_test in range(total_points_train):
		smaller_value = 9999999999
		prototype_smaller = -1
		for item in prototypes_train:
			dist = l2fs_local_norm_train[item][counter_test]
			if dist < smaller_value:
				smaller_value = dist
				prototype_smaller = item
		covered_by_train.append(prototype_smaller)
	return covered_by_train

#########################################################################################################
#Evaluates explanation via prototypes

#fidelity
def calculate_fidelity():
	fidelity = []
	for x in range(0, total_points_test):
		fidelity.append(distance_fidelity[covered_by[x]][x])
	fidelity_median = np.median(fidelity)

	return fidelity_median

#stability
def calculate_stability():
	stability = []
	for x in range(0, total_points_test):
		stability.append(distance_stability[int(covered_by[x])][x])
	stability_median = np.median(stability)

	return stability_median

def evaluate_prototypes():
	fidelity_median = calculate_fidelity()
	stability_median = calculate_stability()
	
	stability_norm = normalize_stability_train_test_minMax(stability_median)
	fidelity_norm = normalize_fidelity_train_test_minMax(fidelity_median)
	gfs = ((1/math.log10(stability_norm))** 2 + (1/math.log10(fidelity_norm))**2) ** 0.5
	
	return gfs

######################################################################################
def calculate_rmse_prototypes(generator_prototypes):  #check whether the regression using only the prototype instances approaches the original model
	cont_rep = 0
	rmse_prototypes = []
	seeds_prototypes = [76, 505, 421, 863, 874, 771, 648, 68, 324, 501, 9759,3077,7947,5460,6171,8794,8271,5087,568,2909,6093,9025,9329,8409,5910,4629,9221,479,4225,1311]
	total_repetitions = len(seeds_prototypes)

	X_train_prototypes = []
	y_train_prototypes = []
	for prot in prototypes:
		X_train_prototypes.append(X_train[prot])
		y_train_prototypes.append(y_train[prot])

	while cont_rep < total_repetitions:
		regr_prototypes = RandomForestRegressor(random_state=seeds_prototypes[cont_rep],n_estimators = 10)
		regr_prototypes.fit(X_train_prototypes, y_train_prototypes)

		y_predicted_prototypes = regr_prototypes.predict(X_test)

		rmse_test_rf_prototypes = np.sqrt(np.mean((y_predicted_prototypes-y_predicted_rf)**2))	#I compare with the initially predicted after all I want to explain the model and not the initial data
		rmse_prototypes.append(rmse_test_rf_prototypes)

		cont_rep += 1

	median_rmse_prototypes = np.median(rmse_prototypes)
	return median_rmse_prototypes

#########################################################################################################
#########################################################################################################

nameDataSetTrain = sys.argv[1]	
nameDataSetTest = sys.argv[2]
namePrototypesProtoDash = sys.argv[3]
plotChart = sys.argv[4] #0: do not plot graph; 1: plot graph

initial = time.time()

training = genfromtxt(nameDataSetTrain, delimiter=',')
test = genfromtxt(nameDataSetTest, delimiter=',')
dimensao = (test.shape[1]-1)

training_original = training
test_original = test

#normalize dados
sc = StandardScaler()  
training = sc.fit_transform(training)  
test = sc.transform(test)

# Training samples
X_train = training[:,:dimensao]
y_train = np.ravel(training[:,dimensao:])

# Testing samples
X_test = test[:,:dimensao]
y_test = np.ravel(test[:,dimensao:])

total_points_train= len(y_train)
total_points_test= len(y_test)

f = open(namePrototypesProtoDash, 'r')
linhaprototypes = f.readline()
linhaprototypes = (linhaprototypes.strip("\n")).split(",")
prototypesProtoDash = []
for item in linhaprototypes:
	prototypesProtoDash.append(int(item))
print(prototypesProtoDash)

#calculate the distance between train/test instances based on attributes
distance_stability = np.zeros(shape=(total_points_train,total_points_test))
for x in range(0, total_points_train):
	for y in range(0, total_points_test):
		distance_stability[x][y] = calculatete_distance_minkowski_train_test(x, y)

		if distance_stability[x][y] > max_stability_train_test:
			max_stability_train_test = distance_stability[x][y]
		if distance_stability[x][y] < min_stability_train_test:
			min_stability_train_test = distance_stability[x][y]

#distance only between train instances
distance_instancias_train = np.zeros(shape=(total_points_train,total_points_train))
for x in range(0, total_points_train):
	for y in range(x, total_points_train):
		distance_calculateda = calculatete_distance_minkowski_train_train(x, y)
		distance_instancias_train[x][y] = distance_calculateda
		distance_instancias_train[y][x] = distance_calculateda

		if distance_calculateda > max_stability_train:
			max_stability_train = distance_calculateda
		if distance_calculateda < min_stability_train:
			min_stability_train = distance_calculateda

strategies = [ "random",
		"medoidsMink03",
		"hub11",
		"GA",
		"ProtoDash",
		"MPEER - s+f",
		"MPEER - gfs+rmse",
		"GA - rmse",
		"MPEER - s+f+rmse"]

total_strategies = int(len(strategies))
total_metrics_evaluation = 1 #GFS
quantity_prototypes = [5, 10] 
for total_prototypes in quantity_prototypes:
	print ("total_prototypes = ",total_prototypes)
	counterRepeats = 0
	
	seeds = [54,4578,152,98,5,321,549,8578,656,6857,8297,5453,6449,2373,9781,4295,2562,2767,8185,8467,9195,2117,2193,8241,720,4387,7813,1947,7196,1072]
	if plotChart == 1:
		limit = 1
	else:
		limit = len(seeds)

	results_executions = np.zeros(shape=(total_strategies,limit,total_metrics_evaluation))

	results_rmse_prototypes = np.zeros(shape=(total_strategies,limit)) 

	while counterRepeats < limit:
		print ("Repetition number = ", counterRepeats)
		seed = seeds[counterRepeats]
		random.seed(seed)
		np.random.seed(seed)

		######################################################################################
		print ("REGRESSAO - RANDOM FOREST: ")

		regr = RandomForestRegressor(random_state=seed,n_estimators = 10)
		regr.fit(X_train, y_train)

		r2_train = regr.score(X_train, y_train)		#Returns the coefficient of determination R^2 of the prediction.
		r2_test = regr.score(X_test, y_test)
		print('R2 no set de train: %.2f' % r2_train)
		print('R2 no set de test: %.2f' % r2_test)

		##y predicted training
		y_predicted_rf_train = regr.predict(X_train)
		
		rmse_test_rf_train = np.sqrt(np.mean((y_predicted_rf_train-y_train)**2))
		print('RMSE no set de training: %.2f' % rmse_test_rf_train)

		##y predicted test
		y_predicted_rf = regr.predict(X_test)

		rmse_test_rf = np.sqrt(np.mean((y_predicted_rf-y_test)**2))
		print('RMSE no set de test: %.2f' % rmse_test_rf)

		######################################################################################
		max_fidelity_train = -9999999999
		max_fidelity_train_test = -9999999999
		min_fidelity_train = 9999999999
		min_fidelity_train_test = 9999999999

		#calculate fidelity and l2fs_local_normalized distance between points
		distance_fidelity = np.zeros(shape=(total_points_train,total_points_test))
		l2fs_local_norm = np.zeros(shape=(total_points_train,total_points_test))
		for x in range(0, total_points_train):
			for y in range(0, total_points_test):
				distance_fidelity[x][y] = ((y_predicted_rf_train[x] - y_predicted_rf[y]) **2)	#my explanation is composed by the attributes of the prototypes plus the actual value of y of each prototype (that is, they are complete instances of the training - not created by the regressor), but my evaluation takes into account the predicted values (I am explaining the regressor and not if is he correct or not)

				if distance_fidelity[x][y] > max_fidelity_train_test:
					max_fidelity_train_test = distance_fidelity[x][y]
				if distance_fidelity[x][y] < min_fidelity_train_test:
					min_fidelity_train_test = distance_fidelity[x][y]

		#normalize
		for x in range(0, total_points_train):
			for y in range(0, total_points_test):
				stability_local_normalized = normalize_stability_train_test_minMax(distance_stability[x][y])
				fidelity_local_normalized = normalize_fidelity_train_test_minMax(distance_fidelity[x][y])
				l2fs_local_norm[x][y] = ((1/math.log10(stability_local_normalized))** 2 + (1/math.log10(fidelity_local_normalized))**2) ** 0.5

		distance_fidelity_train = np.zeros(shape=(total_points_train,total_points_train))
		l2fs_local_norm_train = np.zeros(shape=(total_points_train,total_points_train))

		for x in range(0, total_points_train):
			for y in range(x, total_points_train):
				dist_fid = ((y_predicted_rf_train[x] - y_predicted_rf_train[y]) **2)	#in the calculation of fidelity in the training set I consider only the value predicted by the model
				distance_fidelity_train[x][y] = dist_fid
				distance_fidelity_train[y][x] = dist_fid

				if dist_fid > max_fidelity_train:
					max_fidelity_train = dist_fid
				if dist_fid < min_fidelity_train:
					min_fidelity_train = dist_fid

		#normalize
		for x in range(0, total_points_train):
			for y in range(x, total_points_train):
				fidelity_local_normalized_train = normalize_fidelity_train_minMax(distance_fidelity_train[x][y])
				stability_local_normalized_train = normalize_stability_train_minMax(distance_instancias_train[x][y])
	
				dist_ldfs_local = ((1/math.log10(stability_local_normalized_train))** 2 + (1/math.log10(fidelity_local_normalized_train))**2) ** 0.5
				l2fs_local_norm_train[x][y] = dist_ldfs_local
				l2fs_local_norm_train[y][x] = dist_ldfs_local

		#for kmedoids it is necessary that the distance from a point to itself is zero
		for x in range(0, total_points_train):
			l2fs_local_norm_train[x][x] = 0

		######################################################################################
		final = time.time()
		print("Time memory and initial processing: ", final-initial)
		counterStrategy = 0 
		total_graphics = str(total_prototypes)+"_"+str(counterRepeats)
		print ("")
		
		print ("###CHOICE: random")
		ini = time.time()

		counterRandom = 0
		gfs_best = 9999999999
		prototypes_bestes = []
		while counterRandom < (300*75):
			prototypes = prototypes_random()
			covered_by = assign_l2fs(prototypes)
			gfs = evaluate_prototypes()
			if(gfs < gfs_best):
				gfs_best = gfs
				prototypes_bestes = prototypes
			counterRandom += 1

		prototypes = prototypes_bestes
		fim = time.time()
		print("Time random: ", fim-ini)
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("random")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1

		print ("###CHOICE: MEDOIDS_MINK03")
		prototypes = prototypes_medoids_mink03()
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("medoidsMink03")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1

		print ("###CHOICE: HUB - 11")
		prototypes = prototypes_hubs(11)
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("Hubs")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1

		print ("###CHOICE: GA")
		prototypes = prototypes_ga("gfs")
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("GA")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1

		print ("###CHOICE: ProtoDash")
		prototypes = prototypesProtoDash[0:total_prototypes]
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("ProtoDash")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1
		
		print ("###CHOICE: GA SPEA2 - s+f")
		ini = time.time()
		prototypes = prototypes_MPEER_est_fid(total_graphics, plotChart)
		fim = time.time()
		print("Time CHOICE spea2(est,fid): ", fim-ini)
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("GA SPEA2 - s+f")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1

		print ("###CHOICE: GA SPEA2 - gfs+rmse")
		ini = time.time()
		prototypes = prototypes_MPEER_gfs_rmse(total_graphics, plotChart)
		fim = time.time()
		print("Time CHOICE spea2(gfs,rmse): ", fim-ini)
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("GA SPEA2 - gfs+rmse")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1

		print ("###CHOICE: GA - rmse")
		prototypes = prototypes_ga("rmse")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("GA - rmse")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1

		print ("###CHOICE: GA SPEA2 - s+f+rmse")
		ini = time.time()
		prototypes = prototypes_MPEER_e_f_rmse(total_graphics, plotChart)
		fim = time.time()
		print("Time CHOICE spea2(est,fid,rmse): ", fim-ini)
		print ("chosen prototypes:")
		for itemp in prototypes:
			print (itemp,", ", end="")
		print("\n")
		results_rmse_prototypes[counterStrategy][counterRepeats] = calculate_rmse_prototypes("GA SPEA2 - s+f+rmse")
		#attribution
		covered_by = assign_l2fs(prototypes)
		gfs = evaluate_prototypes()
		results_executions[counterStrategy][counterRepeats] = gfs
		print(results_executions[counterStrategy][counterRepeats])
		print(results_rmse_prototypes[counterStrategy][counterRepeats])
		counterStrategy += 1
		
		counterRepeats += 1

	print ("- Medians ",total_prototypes," prototypes:")
	print ("GFS:")
	aux = 0
	for strategy in strategies:
		print (strategy,", ", end = '')
		for item in (np.around(np.median(results_executions[aux], axis = 0), decimals=4)):
			print (item,", ", end = '')
		print ("")
		aux += 1 
	print ("")
	
	print ("RMSE:")
	aux = 0
	for strategy in strategies:
		print (strategy,", ", (np.around(np.median(results_rmse_prototypes[aux]), decimals=4)))
		aux += 1
	print ("")

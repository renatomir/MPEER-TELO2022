###################################################################################
#Example to run: python3 -u ProtoDash.py ../DataSet/TRAIN_cholesterol > prototypesProtodash_cholesterol
#argv[1]: training data set
###################################################################################

from __future__ import print_function
import numpy as np
from cvxopt.solvers import qp
from cvxopt import matrix, spmatrix
from numpy import array, ndarray
from scipy.spatial.distance import cdist
from qpsolvers import solve_qp
import abc
import sys
import unittest
import os
import pandas as pd
import csv


# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
	ABC = abc.ABC
else:
	ABC = abc.ABCMeta(str('ABC'), (), {})


class DIExplainer(ABC):
	"""
	DIExplainer is the base class for Directly Interpretable unsupervised explainers (DIE).
	Such explainers generally rely on unsupervised techniques to explain datasets and model predictions.
	Examples include DIP-VAE[#1]_, Protodash[#2]_, etc.

	References:
		.. [#1] Variational Inference of Disentangled Latent Concepts from Unlabeled Observations (DIP-VAE), ICLR 2018.
		 Kumar, Sattigeri, Balakrishnan. https://arxiv.org/abs/1711.00848
		.. [#2] ProtoDash: Fast Interpretable Prototype Selection, 2017.
		Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi.
		https://arxiv.org/abs/1707.01212
	"""

	def __init__(self, *argv, **kwargs):
		"""
		Initialize a DIExplainer object.
		ToDo: check common steps that need to be distilled here.
		"""

	@abc.abstractmethod
	def set_params(self, *argv, **kwargs):
		"""
		Set parameters for the explainer.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def explain(self, *argv, **kwargs):
		"""
		Explain the data or model.
		"""
		raise NotImplementedError

################################################################################################

def runOptimiser(K, u, preOptw, initialValue, maxWeight=10000):
	"""
	Args:
		K (double 2d array): Similarity/distance matrix
		u (double array): Mean similarity of each prototype
		preOptw (double): Weight vector
		initialValue (double): Initialize run
		maxWeight (double): Upper bound on weight

	Returns:
		Prototypes, weights and objective values
	"""
	d = u.shape[0]
	lb = np.zeros((d, 1))
	ub = maxWeight * np.ones((d, 1))
	x0 = np.append( preOptw, initialValue/K[d-1, d-1] )

	G = np.vstack((np.identity(d), -1*np.identity(d)))
	h = np.vstack((ub, -1*lb))

	#	 Solve a QP defined as follows:
	#		 minimize
	#			 (1/2) * x.T * P * x + q.T * x
	#		 subject to
	#			 G * x <= h
	#			 A * x == b
	sol = solve_qp(K, -u, G, h, A=None, b=None, solver='cvxopt', initvals=x0)

	# compute objective function value
	x = sol.reshape(sol.shape[0], 1)
	P = K
	q = - u.reshape(u.shape[0], 1)
	obj_value = 1/2 * np.matmul(np.matmul(x.T, P), x) + np.matmul(q.T, x)
	return(sol, obj_value[0,0])



def get_Gaussian_Data(nfeat, numX, numY):
	"""
	Args:
		nfeat (int): Number of features
		numX (int): Size of X
		numY (int): Size of Y

	Returns:
		Datasets X and Y
	"""
	np.random.seed(0)
	X = np.random.normal(0.0, 1.0, (numX, nfeat))
	Y = np.random.normal(0.0, 1.0, (numY, nfeat))

	for i in range(numX):
		X[i, :] = X[i, :] / np.linalg.norm(X[i, :])

	for i in range(numY):
		Y[i, :] = Y[i, :] / np.linalg.norm(Y[i, :])

	return(X, Y)


# expects X & Y in (observations x features) format

def HeuristicSetSelection(X, Y, m, kernelType, sigma):
	"""
	Main prototype selection function.

	Args:
		X (double 2d array): Dataset to select prototypes from
		Y (double 2d array): Dataset to explain
		m (double): Number of prototypes
		kernelType (str): Gaussian, linear or other
		sigma (double): Gaussian kernel width

	Returns:
		Current optimum, the prototypes and objective values throughout selection
	"""
	numY = Y.shape[0]
	numX = X.shape[0]
	allY = np.array(range(numY))

	# Store the mean inner products with X
	if kernelType == 'Gaussian':
		meanInnerProductX = np.zeros((numY, 1))
		for i in range(numY):
			Y1 = Y[i, :]
			Y1 = Y1.reshape(Y1.shape[0], 1).T
			distX = cdist(X, Y1)
			meanInnerProductX[i] = np.sum( np.exp(np.square(distX)/(-2.0 * sigma**2)) ) / numX
	else:
		M = np.dot(Y, np.transpose(X))
		meanInnerProductX = np.sum(M, axis=1) / M.shape[1]

	# move to features x observation format to be consistent with the earlier code version
	X = X.T
	Y = Y.T

	# Intialization
	S = np.zeros(m, dtype=int)
	setValues = np.zeros(m)
	sizeS = 0
	currSetValue = 0.0
	currOptw = np.array([])
	currK = np.array([])
	curru = np.array([])
	runningInnerProduct = np.zeros((m, numY))

	while sizeS < m:

		remainingElements = np.setdiff1d(allY, S[0:sizeS])

		newCurrSetValue = currSetValue
		maxGradient = 0

		for count in range(remainingElements.shape[0]):

			i = remainingElements[count]
			newZ = Y[:, i]

			if sizeS == 0:

				if kernelType == 'Gaussian':
					K = 1
				else:
					K = np.dot(newZ, newZ)

				u = meanInnerProductX[i]
				w = np.max(u / K, 0)
				incrementSetValue = -0.5 * K * (w ** 2) + (u * w)

				if (incrementSetValue > newCurrSetValue) or (count == 1):
					# Bookeeping
					newCurrSetValue = incrementSetValue
					desiredElement = i
					newCurrOptw = w
					currK = K

			else:
				recentlyAdded = Y[:, S[sizeS - 1]]

				if kernelType == 'Gaussian':
					distnewZ = np.linalg.norm(recentlyAdded-newZ)
					runningInnerProduct[sizeS - 1, i] = np.exp( np.square(distnewZ)/(-2.0 * sigma**2 ) )
				else:
					runningInnerProduct[sizeS - 1, i] = np.dot(recentlyAdded, newZ)

				innerProduct = runningInnerProduct[0:sizeS, i]
				if innerProduct.shape[0] > 1:
					innerProduct = innerProduct.reshape((innerProduct.shape[0], 1))

				gradientVal = meanInnerProductX[i] - np.dot(currOptw, innerProduct)

				if (gradientVal > maxGradient) or (count == 1):
					maxGradient = gradientVal
					desiredElement = i
					newinnerProduct = innerProduct[:]

		S[sizeS] = desiredElement

		curru = np.append(curru, meanInnerProductX[desiredElement])

		if sizeS > 0:

			if kernelType == 'Gaussian':
				selfNorm = array([1.0])
			else:
				addedZ = Y[:, desiredElement]
				selfNorm = array( [np.dot(addedZ, addedZ)] )

			K1 = np.hstack((currK, newinnerProduct))

			if newinnerProduct.shape[0] > 1:
				selfNorm = selfNorm.reshape((1,1))
			K2 = np.vstack( (K1, np.hstack((newinnerProduct.T, selfNorm))) )

			currK = K2
			if maxGradient <= 0:
				#newCurrOptw = np.vstack((currOptw[:], np.array([0])))
				newCurrOptw = np.append(currOptw, [0], axis=0)
				newCurrSetValue = currSetValue
			else:
				[newCurrOptw, value] = runOptimiser(currK, curru, currOptw, maxGradient)
				newCurrSetValue = -value

		currOptw = newCurrOptw
		if type(currOptw) != np.ndarray:
			currOptw = np.array([currOptw])

		currSetValue = newCurrSetValue

		setValues[sizeS] = currSetValue
		sizeS = sizeS + 1

	return(currOptw, S, setValues)

################################################################################################

class ProtodashExplainer(DIExplainer):
	"""
	ProtodashExplainer provides exemplar-based explanations for summarizing datasets as well
	as explaining predictions made by an AI model. It employs a fast gradient based algorithm
	to find prototypes along with their (non-negative) importance weights. The algorithm minimizes the maximum
	mean discrepancy metric and has constant factor approximation guarantees for this weakly submodular function. [#]_.

	References:
		.. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
		   "ProtoDash: Fast Interpretable Prototype Selection"
		   <https://arxiv.org/abs/1707.01212>`_
	"""

	def __init__(self):
		"""
		Constructor method, initializes the explainer
		"""
		super(ProtodashExplainer, self).__init__()

	def set_params(self, *argv, **kwargs):
		"""
		Set parameters for the explainer.
		"""
		pass

	def explain(self, X, Y, m, kernelType='other', sigma=2):
		"""
		Return prototypes for data X, Y.

		Args:
			X (double 2d array): Dataset you want to explain.
			Y (double 2d array): Dataset to select prototypical explanations from.
			m (int): Number of prototypes
			kernelType (str): Type of kernel (viz. 'Gaussian', / 'other')
			sigma (double): width of kernel

		Returns:
			m selected prototypes from X and their (unnormalized) importance weights
		"""
		return( HeuristicSetSelection(X, Y, m, kernelType, sigma) )

################################################################################################
dataSet = sys.argv[1]

prototypesN = 100 #selects the 100 most relevant prototypes according to the ProtoDash algorithm

df = pd.read_csv(dataSet, header=None)  
# convert pandas dataframe to numpy
data = df.to_numpy()
np.set_printoptions(suppress=True)

# One-hot encode the data
original = data

# replace nan's with 0's
original[np.isnan(original)] = 0

original = original[:, :-1]		#delete the last column: output value

# Obtain an explanation for data
Y = original		#onehot_encoded
X = Y
explainer = ProtodashExplainer()
# S contains indices of the selected prototypes
# W contains importance weights associated with the selected prototypes 
(W, S, _) = explainer.explain(X, Y, m=prototypesN, kernelType='Gaussian', sigma=2)

# Display the prototypes along with their computed weights
dfs = df.iloc[S].copy()
dfs["Weight"] = W

# Compute normalized importance weights for prototypes
dfs["Weights of Prototypes"] = np.around(W/np.sum(W), 2) 

for item in S:
		print (item,", ", end="")	#prints the selected prototypes
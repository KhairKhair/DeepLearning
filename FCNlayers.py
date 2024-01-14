from numpy import dot, transpose,shape,random,zeros
from random import random as rand
from helper import sigmoid, sigmoid_prime
from layer import Layer


class FcLayer(Layer):
	def __init__(self, num_nodes, next_num_nodes):
		self.weights = random.randn(next_num_nodes, num_nodes) 
		self.biases = random.randn(next_num_nodes, 1) 
		self.DeltaB = zeros(self.biases.shape)
		self.DeltaW = zeros(self.weights.shape)


	def setOutput(self):
		self.output = dot(self.weights, self.input) + self.biases

	def setDoutput(self):
		self.Doutput = dot(self.weights.transpose(), self.Dinput)
		self.DeltaB = self.DeltaB + self.Dinput
		self.DeltaW = self.DeltaW + (dot(self.Dinput, transpose(self.input)))


	def update_params(self, LearningRate):
		self.weights = self.weights - (LearningRate * self.DeltaW)
		self.biases = self.biases - (LearningRate * self.DeltaB)

	def clear_deltas(self):
		self.DeltaB = zeros(self.biases.shape)
		self.DeltaW = zeros(self.weights.shape)


class ActiLayer(Layer):

	def __init__(self, num_nodes, next_num_nodes):
		super().__init__(num_nodes, next_num_nodes)

	def setOutput(self):
		self.output = sigmoid(self.input) 

	def setDoutput(self):
		self.Doutput = self.Dinput * sigmoid_prime(self.input)

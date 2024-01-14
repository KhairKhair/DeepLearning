import numpy as np

def Activation(x):
	return (1/(1+np.exp(x)))

def ActivationPrime(x):
	a = Activation(x)
	return (a * (1-a))

def sigmoid_double(x):
    return (1/(1+np.exp(-x)))

def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)

def sigmoid_double_prime(x):
    return (sigmoid_double(x)*(1-sigmoid_double(x)))

def sigmoid_prime(z):
    return np.vectorize(sigmoid_double_prime)(z)
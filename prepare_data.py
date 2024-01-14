import tensorflow as tf
import numpy as np
import random

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def return_data(numbers):
	Xtrain = []
	Ytrain = []
	for i in range(len(x_train)):
		Xtrain.append(x_train[i].reshape(1,28,28))
		temp = []
		for j in range(10):
			if y_train[i] == j:
				temp.append(1)
			else:
				temp.append(0)
		Ytrain.append(temp)

	TrainIndices = []
	for i in range(len(y_train)):
		label = np.argmax(Ytrain[i])
		if label in numbers:
			TrainIndices.append(i)

	Xtest = []
	Ytest = []
	for i in range(len(x_test)):
		Xtest.append(x_test[i].reshape(1,28,28))
		temp = []
		for j in range(10):
			if y_test[i] == j:
				temp.append(1)
			else:
				temp.append(0)
		Ytest.append(temp)

	TestIndices = []
	for i in range(len(y_test)):
		label = np.argmax(Ytest[i])
		if label in numbers:
			TestIndices.append(i)

	return Xtrain,Ytrain,Xtest,Ytest,TrainIndices,TestIndices


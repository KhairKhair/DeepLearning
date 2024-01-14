from CNNlayers import ConvLayer, ActiConvLayer, StraightenLayer, PoolingLayer
from FCNlayers import FcLayer, ActiLayer 
from prepare_data import return_data
from random import shuffle
import numpy as np
from time import time

class Network():
	def __init__(self, alpha:float):
		self.alpha = alpha
		self.layers = []

	def create_Conv_layer(self, num_k, PoolKSize):
		if len(self.layers) == 0:
			a = ConvLayer((1,28,28), num_k)
		else:
			final = self.layers[-1]
			input_shape = final.output_shape
			a = ConvLayer(input_shape, num_k)

		# not sure of Conv layers are supposed to have activations or not
		#b = ActiConvLayer(a.output_shape)
		b = PoolingLayer(a.output_shape, PoolKSize)

		if (len(self.layers) == 0):
			a.setPrevNext(None, b)
		else:
			self.layers[-1].setNext(a)
			a.setPrevNext(self.layers[-1], b)

		b.setPrevNext(a, None)
#		c.setPrevNext(b, None)
		self.layers.append(a)
		self.layers.append(b)
#		self.layers.append(c)

	def create_straight_layer(self, input_shape):
		a = StraightenLayer(input_shape)
		if (len(self.layers) == 0):
			a.setPrevNext(None,None)
		else:
			self.layers[-1].setNext(a)
			a.setPrevNext(self.layers[-1], None)
		self.layers.append(a)

	def create_dense_layer(self, input, output):
		a = FcLayer(input,output)
		b = ActiLayer(output,output)
		if (len(self.layers) == 0):
			a.setPrevNext(None,b)
			b.setPrevNext(a,None)
		else:
			self.layers[-1].setNext(a)
			a.setPrevNext(self.layers[-1], b)
			b.setPrevNext(a,None)
		self.layers.append(a)
		self.layers.append(b)

	def create_CNN(self, o1,o2, PoolKSize):
		self.create_Conv_layer(o1, PoolKSize)
		self.create_Conv_layer(o2, PoolKSize)
		self.create_straight_layer(self.layers[-1].output_shape)
		self.create_dense_layer(self.layers[-1].output_shape[0], 120)
		self.create_dense_layer(120, 84)
		self.create_dense_layer(84, 10)

	def create_FCN(self, input_shape, o1, o2):
		self.create_straight_layer(input_shape)
		self.create_dense_layer(784, o1)
		self.create_dense_layer(o1, o2)
		self.create_dense_layer(o2,10)


	def train_batch(self, index, batch_size, Xtrain, Ytrain):
		values = Xtrain[index]
		label = Ytrain[index]
		self.layers[0].input = values
		for i in self.layers:
			i.setInput()
			i.setOutput()


		prediction = self.layers[-1].output
		loss_list = []
		for i in range(len(label)):
			v = (prediction[i]-label[i])/batch_size	
			loss_list.append([v[0]])

		self.layers[-1].Dinput = loss_list
		for i in reversed(self.layers):
			i.setDinput()
			i.setDoutput()

		if (index % batch_size) == 0:
			for i in self.layers:
				i.update_params(self.alpha)
				i.clear_deltas()


	def test_batch(self, index, Xtest, Ytest):
		label = Ytest[index]
		values = Xtest[index]
		self.layers[0].input = values
		for i in self.layers:
			i.setInput()
			i.setOutput()
		if np.argmax(self.layers[-1].output) == np.argmax(label):
			return 1
		else:
			return 0




#neural = LeNet(0.1)
#neural.create_layers(4,8,1,84, 2)
##
#Xtrain,Ytrain,Xtest,Ytest,TrainIndices,TestIndices = return_data([0,1,2,3])
#
##for i in pic[0]:
##	for j in i:
##		print(j, end = " ")
##	print()
#
##a = [[[i for i in range(6)] for j in range(6)]]
##print(np.shape(a))
##neural.create_Conv_layer(2,2)
##neural.layers[0].input = a
###neural.layers[0].input = [[[1,2,3,4], [5,4,3,2,1]], [[6,7,8,9,10], [10,9,8,7,6,5]]]
##for i in neural.layers:
##	i.setInput()
##	i.setOutput()
##	print(np.shape(i.input))
##	print(np.shape(i.output))
##
##final = neural.layers[1]
###final.setOutput2()
##for i in final.output:
##	print(i)
##
##
##a = [[i for i in range(1,5)] for i in range(1,5)]
##b = a[::2]
##c = [i[::2] for i in b]
##print(a)
##print(b)
##print(c)
##for i in neural.layers[1].output:
##	for j in i:
##		for x in j:
##			print(x, end = " ")
##		print()
#
#
##print(len(TrainIndices))
##print(len(TestIndices))
##
#epochs = 1000
#batch_size = 1
#t = 500 
#tt = 3000
#for e in range(epochs):
#	t1 = time()
#	print("Epoch ", e, " has begun!")
#	shuffle(TrainIndices)
#	for i1 in range(tt):
#		index = TrainIndices[i1]
#		neural.train_batch(index, batch_size)
#
#	shuffle(TestIndices)
#	wins = 0
#	for i2 in range(t):
#		index = TestIndices[i2]
#		wins += neural.test_batch(index)
#	print(wins/t)
##if (wins/t > 0.95):
##	t = len(TestIndices)
##	neural.alpha = 0.0001
##	tt = int(len(TrainIndices)/2)
##elif (wins/t) < 0.5:
##	neural.alpha = 1		
#	print(t1-time())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#



#for i in range(epochs):
#	print("Epoch ", i, " has begun!")
#	indices = [[np.random.randint(0,50000) for i in range(32)] for i in range(100)]
#	for batch in indices:
#		neural.train_batch(batch)
#
#	indices2 = [np.random.randint(0,10000) for i in range(5000)]
#	wins = 0
#	for i in indices2:
#		wins += neural.test_batch(Ytest[i], Xtest[i])
##	if (wins/2000) > 0.9:
##		a.alpha = 0.1
#	print(wins/5000)

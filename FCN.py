from prepare_data import return_data
from network import Network
from random import shuffle
import numpy as np
from time import time


neural = Network(0.1)

# return_data([0,1,2,3,4,5,6,7,8,9]) would be all of the dataset
Xtrain,Ytrain,Xtest,Ytest,TrainIndices,TestIndices = return_data([0,1,2,3])
neural.create_FCN(np.shape(Xtrain[0]), 300,150)


epochs = 1000
batch_size = 1
train_num = 1000 
test_num = 1500

for e in range(epochs):
	t1 = time()
	print("Epoch ", e+1, " has begun!")
	shuffle(TrainIndices)
	for i1 in range(train_num):
		index = TrainIndices[i1]
		neural.train_batch(index, batch_size, Xtrain, Ytrain)

	shuffle(TestIndices)
	wins = 0
	for i2 in range(test_num):
		index = TestIndices[i2]
		wins += neural.test_batch(index, Xtest, Ytest)
	print("accuracy: ", wins/test_num)
	print("time: ", abs(t1-time()))
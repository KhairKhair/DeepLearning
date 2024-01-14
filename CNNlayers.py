from math import ceil, floor
import numpy as np
from scipy import signal
from layer import Layer
from helper import sigmoid, sigmoid_prime

kSize = 2
def strideConv(arr, arr2, s):
    return signal.convolve2d(arr, arr2, mode='valid')[::s, ::s]

def strideConv2(arr, arr2, s):
    return signal.convolve2d(arr, arr2, mode='full')[::s, ::s]


# SetOutput sets output for current class layer.
# setDoutput sets Doutput (Delta output) for current class layer



class ConvLayer(Layer):
	def __init__(self, input_shape, num_k):
		# input shape is (num_pics, height_pics, width_pics)
		self.num_k = num_k
		self.num_pics, self.input_height, self.input_width = input_shape
		self.output_shape = self.num_k, self.input_height-(kSize - 1), self.input_width-(kSize-1)
		# self.num_k decides how many different sets of kernells there will be
		# each of these sets will contian self.num_pics number of kernells
		self.kernells = np.random.randn(num_k, self.num_pics, kSize, kSize)
		self.biases = np.random.randn(*self.output_shape)
		self.DeltaK = np.zeros(np.shape(self.kernells))


	def setOutput(self):
		self.output = np.copy(self.biases)
		for i in range(self.num_k):
			for j in range(self.num_pics):
				self.output[i] += signal.correlate2d(self.input[j], self.kernells[i,j], "valid")


	def setDoutput(self):
		self.Doutput = np.zeros(np.shape(self.input))
		for i in range(self.num_k):
			for j in range(self.num_pics):
				self.DeltaK[i,j] = signal.correlate2d(self.input[j], self.Dinput[i], "valid")	
				self.Doutput[j] = signal.correlate2d(self.Dinput[i], self.kernells[i,j], "full")	

	def update_params(self, alpha):
		self.kernells = self.kernells - (self.DeltaK * alpha) 	
		self.biases = self.biases - (self.Dinput * alpha) 	

	def clear_deltas(self):
		self.DeltaK = np.zeros(np.shape(self.kernells))


class ActiConvLayer(Layer):
	def __init__(self, input_shape):
		self.num_pics, self.input_height, self.input_width = input_shape
		self.output_shape = input_shape


	def setOutput(self):
		#self.output = []
		#for i in self.input:
		#	self.output.append(sigmoid(i))
		#print(np.shape(self.output))
		self.output = sigmoid(self.input)

	def setDoutput(self):
		#self.Doutput = []
		#for i in self.Dinput:
		#	self.Doutput.append(sigmoid_prime(i))
		sig = sigmoid(self.Dinput)
		self.Doutput = sig * (1-sig)





class StraightenLayer(Layer):
	def __init__(self, input_shape):
		self.num_pics, self.input_height, self.input_width = input_shape
		self.input_shape = input_shape
		self.output_shape = (self.num_pics * self.input_height * self.input_height, 1)

	def setOutput(self):
		self.output = self.input.reshape(*self.output_shape)

	def setDoutput(self):
		self.Doutput = self.Dinput.reshape(*self.input_shape)


class PoolingLayer(Layer):
	def __init__(self, input_shape, kSize):
		# input shape is (num_pics, height_pics, width_pics)
		self.num_pics, self.input_height, self.input_width = input_shape
		self.num_k = self.num_pics
		self.output_shape = self.num_k, self.input_height-(kSize - 1), self.input_width-(kSize-1)
		# self.num_k decides how many different sets of kernells there will be
		# each of these sets will contian self.num_pics number of kernells
		self.kernells = [[(1/kSize)**2 for i in range(kSize)] for j in range(kSize)]



	def setOutput(self):
		self.output = np.zeros(self.output_shape)
		for i in range(self.num_k):
			for j in range(self.num_pics):
				self.output[i] += signal.correlate2d(self.input[j], self.kernells, "valid")

	def setDoutput(self):
		self.Doutput = np.zeros(np.shape(self.input))
		for i in range(self.num_k):
			for j in range(self.num_pics):
				self.Doutput[j] = signal.correlate2d(self.Dinput[i], self.kernells, "full")	


# attempt at building poolinglayer with stride
#class PoolingLayer(Layer):
#	def __init__(self, input_shape, kSize, stride = 0):
#		# input shape is (num_pics, height_pics, width_pics)
#		if stride < kSize:
#			self.stride = kSize
#		else:
#			self.stride = kSize
#		self.num_pics, self.input_height, self.input_width = input_shape
#		self.num_k = self.num_pics
#		self.output_shape = self.num_k, ceil((self.input_height-(kSize - 1))/self.stride), ceil(((self.input_width-(kSize-1))/self.stride))
#		# self.num_k decides how many different sets of kernells there will be
#		# each of these sets will contian self.num_pics number of kernells
#		self.kernells = [[(1/kSize)**2 for i in range(kSize)] for j in range(kSize)]
#		self.r = 1/kSize
#
#	def setOutput(self):
#		self.output = np.zeros(self.output_shape)
#		for i in range(self.num_k):
#			for j in range(self.num_pics):
#				value = 0
#				pic = self.input[j]	
#				for y in range(0,self.input_height,self.stride):
#					if self.input_height-y < self.stride:
#						break
#					for x in range(0,self.input_width, self.stride):
#						if  self.input_width - x < self.stride:
#							break
#						value = 0
#						for w in range(self.stride):
#							for h in range(self.stride):
#								value += self.r * pic[y+h][x+w]
#						self.output[i][y][x] += value

	#def setOutput(self):
	#	self.output = np.zeros(self.output_shape)
	#	#print(np.shape(self.input))
	#	#print(self.output_shape)
	#	#print(np.shape(self.input))
	#	for i in range(self.num_k):
	#		for j in range(self.num_pics):
	#			a = signal.correlate2d(self.input[j], self.kernells, "valid")
	#			a = a[::self.stride]
	#			a = [i[::self.stride] for i in a]
	#			self.output[i] += a
#				self.output[i] += signal.correlate2d(self.input[j], self.kernells, "valid")
#		a = [self.output[i][::self.stride] for i in range(self.num_k)]
#		b = [[i[::self.stride] for i in j] for j in a]
#		self.output = b
#		print(self.stride)

	#def setOutput(self):
	#	self.output = np.zeros(self.output_shape)
	#	for i in range(self.num_k):
	#		for j in range(self.num_pics):
	#			pic = self.input[j]
	#			for y in range(self.input_height):
	#				for x in range(self.input_width):
	#					index = floor(x/self.stride)
	#					indexY = floor(y/self.stride)
	#					self.output[i][indexY][index] += self.r * pic[y][x]

	#def setDoutput(self):
	#	self.Doutput = []
	#	#self.Doutput = np.zeros(np.shape(self.input))
	#	#print(np.shape(self.input))
	#	#print(np.shape(self.output))
	#	#print(np.shape(self.Dinput))
	#	#for i in range(self.num_k):
	#	#	for j in range(self.num_pics):
	#	#		print(np.shape(signal.correlate2d(self.Dinput[i], self.kernells, "full")))
	#	#		self.Doutput[j] = signal.correlate2d(self.Dinput[i], self.kernells, "full")	
	#	#for i in range(self.num_k):
	#	#	for j in range(self.num_pics):
	#			#self.Doutput[j] = []
	#	print("000000000000000000000000000000000000000")
	#	print(np.shape(self.input))
	#	print(self.output_shape)
	#	for pic in self.Dinput:
	#		p = []
	#		for j in range(self.output_shape[1]):
	#			new_row = [[self.r * v for i in range(self.stride)] for v in pic[j]]
	#			new_row = np.reshape(new_row, self.stride * len(pic[j]))
	#			for i in range(self.stride):
	#				p.append(new_row)
	#		self.Doutput.append(p)

	#	print(np.shape(self.Doutput))
	#	print("000000000000000000000000000000000000000")







#def setDoutput(self):
#	self.Doutput = np.zeros(np.shape(self.input))
#	for i in range(self.num_k):
#		for j in range(self.num_pics):









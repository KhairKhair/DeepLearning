
class Layer():
	def __init__(self, num_nodes, next_num_nodes):
		self.weights = []
		self.biases = []
		self.input = []
		self.output = []
		self.Dinput = []
		self.Doutput = []


	def setPrevNext(self, prev,next):
		self.prev = prev	
		self.next = next

	def setPrev(self, prev):
		self.prev = prev

	def setNext(self, next):
		self.next = next


	def setOutput(self):
		#placeholder
		return

	def setDoutput(self):
		#placeholder
		return

	def update_params(self, LearningRate):
		return

	def clear_deltas(self):
		return

	def setInput(self):
		if self.prev == None:
			return
		else:
			self.input = self.prev.output

	def setDinput(self):
		if self.next == None:
			return
		else:
			self.Dinput = self.next.Doutput

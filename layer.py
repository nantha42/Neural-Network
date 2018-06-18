import numpy as np 
class unit:
	def __init__(self,X=0,grad=0):
		self.val = X
		self.grad = 0

class multiply:
	def __init__(self,u1=None):
		self.u0 = None#previous activation
		self.u1 = u1#weights
		self.u2 = unit(0,0)

	def set(self,u0,u1):
		self.u0 = u0 
		self.u1 = u1

	def forward(self):
		
		#self.u2 = unit(X= np.dot(self.u0.val,self.u1.val),grad = 0)
		self.u2.val = np.dot(self.u0.val,self.u1.val)
		return self.u2

	def backward(self):
		self.u0.grad = self.u1.val*self.u2.grad 
		self.u0.grad = np.dot(self.u2.grad,self.u1.val.T)
		self.u1.grad = np.dot(self.u0.val.T,self.u2.grad) 

class add:
	def __init__(self,u1 = None):
		self.u0 = None
		self.u1 = u1
		self.u2 = unit(0,0)
	def set(self,u0,u1):
		self.u0 = u0 
		self.u1 = u1

	def forward(self):
		#self.u2 = unit(X = self.u0.val+self.u1.val,grad = 0)
		self.u2.val = self.u0.val+self.u1.val
		return self.u2

	def backward(self):
		self.u0.grad = 1*self.u2.grad 
		self.u1.grad = 1*self.u2.grad 

class activation:
	def __init__(self,typ=None):
		self.u0 = None
		self.type = typ
		self.u1 = unit(0,0)

	def set(self,u0,typ):
		self.u0 = u0
		self.type = typ

	def forward(self):
		if(self.type == 'sigmoid'):
			val = 1/(1+np.exp(-self.u0.val))
			#self.u1 = unit(val,0)
			self.u1.val = val 
			return self.u1
		elif(self.type == 'relu'):
			val = (self.u0.val>0)*self.u0.val
			#self.u1 = unit(val)
			self.u1.val = val 
			return self.u1

	def backward(self):
		if(self.type == 'sigmoid'):
			self.u0.grad = self.u1.val*(1-self.u1.val)*self.u1.grad
		elif(self.type == 'relu'):
			self.u0.grad = (self.u1.val>0)*self.u1.grad

class loss:
	def __init__(self):
		self.u0 = None 
		self.u1 = None
		self.u2 = unit(0,0)

	def set(self,u0,u1):
		self.u0 = u0 
		self.u1 = u1

	def forward(self):
		val = .5*(self.u0.val-self.u1.val)**2
		#self.u2 = unit(X = val,grad= 0)
		self.u2.val = val
		return self.u2 

	def backward(self):
		self.u0.grad = (self.u0.val-self.u1.val)*1

class layer:
	def __init__(self,weightshape=(1,1),act_func = 'relu'):
		w = 2*np.random.random(weightshape)-1
		self.w = unit(w,0)
		b = np.random.random((1,weightshape[1]))
		self.b = unit(b,0)
		self.propagatemap = []


		m = multiply()
		m.u1 = self.w 
		self.propagatemap.append(m)
		s = add()
		s.u0 = m.u2 
		s.u1 = self.b 
		self.propagatemap.append(s)
		o = activation(typ=act_func)
		o.u0 = s.u2 
		self.propagatemap.append(o)

	def forward(self,x):
		self.propagatemap[0].u0 = x
		for i in range(3):
			self.propagatemap[i].forward()
		return self.propagatemap[2].u1

	def calculateloss(self,y):
		o = self.propagatemap[2].u1
		l = loss()
		l.set(o,y)
		l.forward()
		self.propagatemap.append(l)
		return self.propagatemap[3].u2
		pass

	def backward(self):
		length = len(self.propagatemap)
		index = [x for x in range(len(self.propagatemap))]
		index.reverse()
		
		for i in index:
			self.propagatemap[i].backward()
		

	def apply_gradient(self):
		self.w.val -= self.w.grad
		self.b.val -= self.b.grad 
		pass

class denseLayers:
	def __init__(self):
		self.layers = []

	def add(self,layer):
		self.layers.append(layer)

	def newlayer(self,shape=None,activation=None):
		self.add(layer(shape,activation))

	def train(self,X,Y,epochs):
		nlayer = len(self.layers)
		X = unit(X,0)
		Y = unit(Y,0)
		for epoch in range(epochs*10):
			x = X
			i=0
			for i in range(nlayer):
				x = self.layers[i].forward(x)
			self.layers[i].calculateloss(Y)
			for i in range(1,nlayer+1):
				self.layers[-i].backward()
			for i in range(nlayer):
				self.layers[i].apply_gradient()

	def predict(self,x):
		x = unit(x,0)
		for i in range(len(self.layers)):
			x = self.layers[i].forward(x)
		return x.val

"""
if __name__ == '__main__':
	np.random.seed(1)
	{
	inp	 = np.array([[1,1,0]])
	out = np.array([0,1])
	W = np.random.random((3,2))
	B = np.random.random([1,2])


	X = unit(inp)
	print(inp.shape)
	w = unit(W)
	b = unit(B)
	y = unit(out)
	}	
	for i in range(100):
		m = multiply()
		m.set(X,w)
		s = add()
		s.set(m.forward(),b)
		o = activation()
		o.set(s.forward(),typ = "sigmoid")
		los = loss()
		los.set(o.forward(),y)
		final = los.forward()
		#print(final.val)
		final.grad = 1

		los.backward()
		o.backward()
		s.backward()
		m.backward()
		w.val -= w.grad 
		b.val -= b.grad
		print(o.u1.val)

"""
"""
if __name__ == '__main__':
	np.random.seed(1)

	inp	 = np.array([[1,1,0]])
	out = np.array([0,1])
	W = np.random.random((3,2))
	B = np.random.random([1,2])


	X = unit(inp)
	print(inp.shape)
	w = unit(W)
	b = unit(B)
	y = unit(out)

	l1 = layer((3,2),"sigmoid")
	for i in range(100):
		l1.forward(X)
		l1.calculateloss(y)
		l1.backward()
		l1.apply_gradient()

	print(l1.forward(X).val)
"""
if __name__ == '__main__':
	np.random.seed(1)

	inp	 = np.array([[1,1,0]])
	out = np.array([0,1])
	W = np.random.random((3,2))
	B = np.random.random([1,2])


	nn = denseLayers()
	nn.newlayer((3,3),"relu")
	nn.newlayer((3,2),"sigmoid")

	nn.train(inp,out,100)
	print(nn.predict(inp))
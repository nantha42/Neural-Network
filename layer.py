import numpy as np 
class unit:
	def __init__(self,X=0,grad=0):
		self.val = X
		self.grad = 0

class multiply:
	def __init__(self,x,y):
		self.u0 = x#previous activation
		self.u1 = y#weights

	def forward(self):
		self.u2 = unit(X= np.dot(self.u0.val,self.u1.val),grad = 0)
		return self.u2

	def backward(self):
		self.u0.grad = self.u1.val*self.u2.grad 
		self.u0.grad = np.dot(self.u2.grad,self.u1.val.T)
		self.u1.grad = np.dot(self.u0.val.T,self.u2.grad) 

class add:
	def __init__(self,x,y):
		self.u0 = x
		self.u1 = y
	def forward(self):
		self.u2 = unit(X = self.u0.val+self.u1.val,grad = 0)
		return self.u2
	def backward(self):
		self.u0.grad = 1*self.u2.grad 
		self.u1.grad = 1*self.u2.grad 

class activation:
	def __init__(self,x,typ):
		self.u0 = x 
		self.type = typ

	def forward(self):
		if(self.type == 'sigmoid'):
			val = 1/(1+np.exp(-self.u0.val))
			self.u1 = unit(val,0)
			return self.u1
		elif(self.type == 'relu'):
			val = (self.u0.val>0)*self.u0.val
			self.u1 = unit(val)
			return self.u1

	def backward(self):
		if(self.type == 'sigmoid'):
			self.u0.grad = self.u1.val*(1-self.u1.val)*self.u1.grad
		elif(self.type == 'relu'):
			self.u0.grad = (self.u1.val>0)*self.u1.grad

class loss:
	def __init__(self,out,exp):
		self.u0 = out
		self.u1 = exp 

	def forward(self):
		val = .5*(self.u0.val-self.u1.val)**2
		self.u2 = unit(X = val,grad= 0)
		return self.u2 

	def backward(self):
		self.u0.grad = (self.u0.val-self.u1.val)*1

np.random.seed(1)

inp = np.array([[1,1,0]])
out = np.array([0,1])
W = np.random.random((3,2))
B = np.random.random([1,2])


X = unit(inp)
print(inp.shape)
w = unit(W)
b = unit(B)
y = unit(out)

for i in range(100):
	m = multiply(X,w)
	s = add(m.forward(),b)
	o = activation(s.forward(),typ = "relu")
	los = loss(o.forward(),y)
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










import numpy as np 

def lin(x,deriv=False):
	if(deriv==True):
	    return (x>0)*1

	return (x>0)*x

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0],[1],[1],[0]])

np.random.seed(1)
w1 = np.random.random((3,6))
w2 = np.random.random((6,5))
w3 = np.random.random((5,1))

for i in range(1000):
	a0 = X
	z1 = np.dot(a0,w1)
	a1 = lin(z1)

	z2 = np.dot(a1,w2)
	a2 = lin(z2)

	z3 = np.dot(a2,w3)
	a3 = nonlin(z3)

	C = Y-a3
	#if(i%10==0):
		#print(np.mean(np.abs(C)))

	Cdz3 = C*nonlin(a3,True)
	#print(C,z2,Cdz2)
	Cdz2 = Cdz3.dot(w3.T)*lin(a2,True)
	Cdz1 = Cdz2.dot(w2.T)*lin(a1,True)
	
	w3 += a2.T.dot(Cdz3)	
	w2 += a1.T.dot(Cdz2)
	w1 += a0.T.dot(Cdz1)
	
#print(w1)	
#print(a3)
print(np.mean(np.abs(C)))
#0.0044622369550180365
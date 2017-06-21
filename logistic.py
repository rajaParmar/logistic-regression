import numpy as np
import matplotlib.pyplot as plt 
import math

def make_vector(list_val):
	return np.array([list_val],dtype=np.float).T


filename="ex2data1.txt"
raw=open(filename,"rt")
data=np.loadtxt(raw,delimiter=",")
theta=make_vector([0.5,0.8,0.1])
x1=data[0:80,0]
x2=data[0:80,1]
y=data[0:80,2]
x3=data[81:99,0]
x4=data[81:99,1]
y2=data[81:99,2]

def plot_data():
	for i in data:
		if(i[2]==1.0):
			plt.plot(i[0],i[1],'bo')
		else: plt.plot(i[0],i[1],'r+')
	plt.show()

def sigmoid(x):
	return 1/(1+np.exp(-x))

def hypothesis(theta,x):
	return sigmoid((theta.T@x)[0])[0]


def cost(theta):
	temp=0
	for i in range(0,x1.shape[0]):
		x=make_vector([1,x1[i],x2[i]])
		print(hypothesis(theta,x),y[i])
		
		temp=temp+(-y[i]*math.log(hypothesis(theta,x)))-((1-y[i])*math.log(1-hypothesis(theta,x)))
	temp=temp/x1.shape
	return temp

def cal_accuracy():
	correct=0
	for i in range(0,x3.shape[0]):
		if(hypothesis(theta,make_vector([1,x3[i],x4[i]]))==y2[i]):
			correct=correct+1
	return correct/19

#print(data)


#print(x1.shape)


print(cost(theta))
#print(cal_accuracy())
#print(math.log(hypothesis(make_vector([1,2,3]),make_vector([1,2,4]))))
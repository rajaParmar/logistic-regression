import numpy as np
import matplotlib.pyplot as plt 
import math

def make_vector(list_val):
	return np.array([list_val],dtype=np.float).T


filename="ex2data1.txt"
raw=open(filename,"rt")
data=np.loadtxt(raw,delimiter=",")
theta=make_vector([0,0,0])
x1=data[0:70,0]
x2=data[0:70,1]
y=data[0:70,2]
x3=data[71:99,0]
x4=data[71:99,1]
y2=data[71:99,2]

def plot_data():
	for i in data:
		if(i[2]==1.0):
			plt.plot(i[0],i[1],'bo')
		else: plt.plot(i[0],i[1],'r+')
	
	plt.plot(x1,x2,eval('theta[0]*1+theta[1]*x1+theta[2]*x2'))
	plt.show()

def sigmoid(x):
	return 1/(1+np.exp(-x))

def hypothesis(theta,x):
	return sigmoid((theta.T@x)[0])[0]


def cost(theta):
	temp=0
	for i in range(0,x1.shape[0]):
		x=make_vector([1,x1[i],x2[i]])
		#print(hypothesis(theta,x),y[i])z
		#print(1-hypothesis(theta,x))	
		#print(x)
		#print(i)
		#print(hypothesis(theta,x))
		# if(y[i]==1.0):
		# 	temp=temp+(-math.log(hypothesis(theta,x)))
		# if(y[i]==0.0):
		# 	temp=temp+(-math.log(1-hypothesis(theta,x)))
		#print(y[i])
		temp=temp+(-y[i]*math.log(hypothesis(theta,x)))-((1-y[i])*math.log(1-hypothesis(theta,x)))
	
	temp=temp/x1.shape[0]
	return temp

def cal_accuracy(theta):
	correct=0
	for i in range(0,x3.shape[0]):
		#print(hypothesis(theta,make_vector([1,x3[i],x4[i]])),y2[i])

		if(hypothesis(theta,make_vector([1,x3[i],x4[i]])>=0.5)):
			final_prediction=1
		else:
			final_prediction=0
		if((final_prediction)==y2[i]):
			correct=correct+1
	return correct/x3.shape[0]

def gradient_descent(theta):
	sum=make_vector([0,0,0])
	for i in range(0,x1.shape[0]):
		x=make_vector([1,x1[i],x2[i]])
		h=hypothesis(theta,x)
		temp_vector=x*(h-y[i])
		sum=sum+temp_vector
	sum=sum/x1.shape[0]
	#print(sum)
	theta=theta-(0.0000001*sum)
	#print(theta)
	return theta

for i in range(400):

	theta=gradient_descent(theta)
#print(data)

print(theta)
print(cost(theta))
#print(x1.shape)

#plot_data()
# print(make_vector([1,1,2,4])+make_vector([1,1,2,5]))

# print(cal_accuracy(theta)*100)
#print(hypothesis(make_vector([1,2,3]),make_vector([1,2,4])))
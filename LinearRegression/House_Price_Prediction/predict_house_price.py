import numpy as np 
import pandas as pd 
import random as rd
import matplotlib.pyplot as plt 
import pickle 

df = pd.read_csv('housingdata.csv')
rows, columns = df.shape

X = np.array(df.drop(df.columns[[columns-1]],1))
y = np.array(df[df.columns[-1]])

X = X.tolist()
y = y.tolist()

test_size = 0.2
no_of_iteration = 500
alpha = 1

def test_train_split(X,y,test_size=0.2):
	m = len(X)
	n = int(m*-test_size)
	X_train = X[:n]
	X_test = X[n:]
	y_train = y[:n]
	y_test = y[n:]
	return X_train,X_test,y_train,y_test
def normaliese(X):
	norm=[]
	for i in range(len(X[0])):
		new_list =  [item[i] for item in X]
		u,v = (max(new_list)-min(new_list),sum(new_list)/len(X))
		norm.append((u,v))
	for i in range(len(X)):
		for j in range(len(norm)):
			if j is not 0:
				X[i][j] = (X[i][j]-norm[j][1])/float(norm[j][0])
	return X
		
class MultiLinearRegression:
	theta = []
	def __init__(self):
		self.theta = []
	
	def predict(self,feature):
		h=0;
		for i in range(len(self.theta)):
			h += self.theta[i]*feature[i]
		return h 
	
	def getCost(self,X,y):
		cost = 0;
		for i in range(len(X)):
			cost += (self.predict(X[i])-y[i])**2
		return cost
	def fit(self,X,y,no_of_iteration=300,alpha=0.1):
		size = len(X)
		no_of_features = len(X[0])
		self.theta=[rd.random() for i in range(no_of_features)]
		for i in range(no_of_iteration):
			new_theta = []
			for j in range(no_of_features):
				sum=0;
				for k in range(size):
					sum += (self.predict(X[k])-y[k])*X[k][j]
				new_theta.append(self.theta[j]-(float(alpha)/size)*sum)
			self.theta = new_theta
			cost = self.getCost(X,y)

X = [[1]+X[i] for i in range(len(X))]
X = normaliese(X)
X_train,X_test,y_train,y_test = test_train_split(X,y,test_size)
#reg = MultiLinearRegression()
#reg.fit(X_train,y_train,no_of_iteration,alpha)
#with open('multiregression.pickle','wb') as f:
#	pickle.dump(reg,f)

pickle_in = open('multiregression.pickle','rb')
reg = pickle.load(pickle_in)
coff = reg.theta

import pandas as pd
import numpy as np 
import math,random,time
from sklearn.utils import shuffle 

class Preprocessing:
	def train_test_split(self,X,y,test_size=0.2):
		m = int(len(X)*test_size)
		X_train = X[:-m]
		X_test = X[-m:]
		y_train = y[:-m]
		y_test = y[-m:]
		return X_train,X_test,y_train,y_test
		
class LogisticRegression:
	coef = []
	def __init__(self):
		self.coef = []
	def getCost(self):
		return self.coef 
	def predict(self,example,p):
		hypo = 0
		for j in range(len(example)):
			hypo += float(example[j])*float(self.coef[j])
		return 1.0/(1+math.exp(-hypo))
	
	def getCost(self,h,y):
		cost = 0
		m = len(h)
		for i in range(len(y)):
			if y[i]==1:
				cost += math.log(1-h[i])
			else:
				cost += math.log(h[i])
		
		return -cost/m
			
	def fit(self,X_train,y_train,num_iter,alpha):
		
		n = len(X_train[0])
		m = len(X_train)
		self.coef = [1 for j in range(n)]
		for iter in range(num_iter):
			temp = []
			for j in range(len(self.coef)):
				sum = 0
				h = []
				for i in range(len(X_train)):
					h.append(self.predict(X_train[i],y_train[i]))
					sum += (-y_train[i])*X_train[i][j]
				temp.append(self.coef[j]-(alpha/m)*sum)
			self.coef = temp 
			cost = self.getCost(h,y_train)
			print cost 
			
	def accuracy(self,feature_set,prediction_set):
		m = len(feature_set)
		#for i in range(len(feature_set)):
			

def main():
	df = pd.read_csv('breast-cancer-wisconsin-data.csv')
	df = shuffle(df)
	df.replace(to_replace='?',value=5,inplace=True)
	rows,columns = df.shape
	X = np.array(df.drop(df.columns[[0,columns-1]],1))
	y = np.array(df[df.columns[-1]])

	X = X.tolist()
	y = y.tolist()

	# 2 refer to benign tumor 4 refer to malignant tumor 
	for j in range(len(y)):
		if y[j]==4:
			y[j]=1
		else:
			y[j]=0
		
	for i in range(len(X)):
		for j in range(len(X[0])):
			X[i][j] = (int(X[i][j])-5)/10.0
		y[i] = int(y[i])
	for i in range(len(X)):
		X[i] = [1] + X[i] 
	
	test_size = 0.2
	num_iter = 300
	alpha = 0.3
	pr = Preprocessing()
	X_train,X_test,y_train,y_test = pr.train_test_split(X,y,test_size)
	lr = LogisticRegression()
	lr.fit(X_train,y_train,num_iter,alpha)
	print lr.getCoef()
	
if __name__ == "__main__":
	main()
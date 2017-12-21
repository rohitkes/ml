# SVM Kernal Implementation 

import numpy as np 
import pandas as pd 
import math
from sklearn import cross_validation

df = pd.read_csv('testout.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
X = X.tolist()
##Add x0 in the X 
#for i in range(len(X)):
#	X[i] = [1]+X[i]


y = y.tolist()	
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

class Svm_Kernal:

	def dist(self,v1,v2):
		sum = 0
		for i in range(len(v1)):
			sum = sum + (float(v1[i])-float(v2[i]))**2
		return sum
	'''
	def similarity(X,L,sigma):
		similarity_matrix = []
		for features in X:
			feature_matrix = [1]
			for landmark in L:
				feature_matrix.append(exp(-1.0/(2*sigma)*dist(features,lanmark)))
			similarity_matrix.append(feature_matrix)
		return similarity_matrix
	'''
	def fit(self,X,y,sigma):
		L = X;
		similarity_matrix = []
		for i in range(len(X)):
			feature_matrix = [1]
			for j in range(len(L)):
				feature_matrix.append(math.exp(-1.0/(2*sigma)*self.dist(X[i],L[j])))
			
			similarity_matrix.append(feature_matrix)
		print len(similarity_matrix[0]),len(X)

clf = Svm_Kernal()
clf.fit(X,y,10)


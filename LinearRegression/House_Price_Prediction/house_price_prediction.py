from sklearn.utils import shuffle 
import pandas as pd;
import numpy as np;
import time,random,pickle

mean = []
delta = []
class cross_validation:
	
	def test_train_split(self,featureSet,prediction,test_size=0.2):
			countFeatuteSet = len(featureSet)
			countPredictionSet = len(prediction)
			try:
				if countFeatuteSet!=countPredictionSet:
					raise Exception("Length of the feature set and the prediction set is not same",(countFeatuteSet,countPredictionSet))
		
				test_set_size = int(0.2*countFeatuteSet)
				X_train = featureSet[:-test_set_size]
				X_test = featureSet[-test_set_size:]
				y_train = prediction[:-test_set_size]
				y_test = prediction[-test_set_size:]
				return X_train,X_test,y_train,y_test
				
			except Exception as ex:
				print ex
				
	def normaliese(self,X):
		m = len(X);
		n = len(X[0])
		for i in range(n):
			mean.append(sum([X[j][i] for j in range(m)])/float(m))
		for i in range(n):
			colm = [X[j][i] for j in range(m)]	
			delta.append(max(colm)-min(colm))
			
		for i in range(m):
			for j in range(n):
				X[i][j] = float(X[i][j]-mean[j])/delta[j]
		return X
	def add_bias_unit(self,X):
		for i in range(len(X)):
			X[i] = [1]+X[i]
		return X

class LinearRegression:
		coef = []
		def predict(self,feature):
			return sum([self.coef[j]*feature[j] for j in range(len(self.coef))])
			
		temp = []
		def updateCoef(self,X,y,h,alpha):
			new_coef = []
			m = len(X)
			for j in range(len(self.coef)):
				grad = 0
				for i in range(m):
					grad = grad + (1.0/m)*((h[i]-y[i])*X[i][j])
				new_coef.append(self.coef[j]-alpha*grad)
			self.coef = new_coef
		def predictNewExample(self,feature):
			for j in range(len(mean)):
				feature[j] = (float(feature[j])-mean[j])/delta[j]
			feature = [1] + feature
			return self.predict(feature)
		def fit(self,featureSet,prediction_set,num_iter=300,alpha=0.1):
			m = len(featureSet)
			n = len(featureSet[0])
			self.coef = [0 for i in range(n)]
			for iter in range(num_iter):
				costVal = 0;
				h = []
				for j in range(len(featureSet) ):
					h.append(self.predict(featureSet[j]))
					costVal += ((h[j]-prediction_set[j])**2)/(2*m)
				self.updateCoef(featureSet,prediction_set,h,alpha)
			print "Cost:",costVal
def main():
	df = pd.read_csv('housingdata.csv')
	df = shuffle(df)

	rows,columns = df.shape
	X = np.array(df.drop(df.columns[[columns-1]],1))
	y = np.array(df[df.columns[-1]])

	X = X.tolist()
	y = y.tolist()	
	num_iter = 1000
	alpha = 1
	cr = cross_validation()
	X = cr.normaliese(X)
	X = cr.add_bias_unit(X)
	X_train,X_test,y_train,y_test = cr.test_train_split(X,y)
	#reg = LinearRegression()
	#reg.fit(X_train,y_train,num_iter,alpha)
	#with open('coef.pickle','wb') as f:
	#	pickle.dump(reg,f)

	pickle_in = open('coef.pickle','rb')
	reg = pickle.load(pickle_in)
	new_example = [0.55778,0,21.89,0,0.624,6.335,98.2,2.1107,4,437,21.2,394.67,16.96]
	print reg.predictNewExample(new_example)
if __name__ == "__main__":
	main()
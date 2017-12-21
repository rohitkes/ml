import numpy as np 
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd 
import pickle
df = pd.read_csv('testout.txt')
print len(df)
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#clf = neighbors.KNeighborsClassifier()
#clf.fit(X_train,y_train)

#with open('knn.pickle','wb') as f:
#	pickle.dump(clf,f)

pickle_in = open('knn.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)
print accuracy
example_measure = np.array([4,2,1,1,1,2,3,2,1])
example_measure = example_measure.reshape(1,-1)
prediction = clf.predict(example_measure)
print prediction
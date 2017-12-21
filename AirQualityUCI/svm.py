import numpy as np 
import pandas as pd 
from sklearn import preprocessing, cross_validation, svm 
import pickle 

df = pd.read_csv('testout.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#clf = svm.SVC()
#clf.fit(X_train,y_train)
#with open('svm.pickle','wb') as f:
#	pickle.dump(clf,f)

pickle_in = open('svm.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)
print accuracy

import numpy as np 
import pandas as pd 
import math
import random 
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation 

df = pd.read_csv('testout.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
#full_data = df.astype(float).values.tolist()

df.dropna(inplace=True)
#random.shuffle(df)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf  = LogisticRegression(n_jobs=-1)
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print clf.coef_,clf.intercept_
from sklearn import datasets,svm
from sklearn.externals import joblib

import pickle
iris = datasets.load_iris()
digits = datasets.load_digits()
#print digits.data[:10]
#print digits.target[:-1]
clf = svm.SVC(gamma=0.001,C=100.)
clf.fit(digits.data[:-1],digits.target[:-1])
clf.predict(digits.data[-1:])

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print clf2.predict(digits.data[-1:])
joblib.dump(clf,'digit.pkl')
clf = joblib.load('digit.pkl')
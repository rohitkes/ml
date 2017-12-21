'''from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[10,12],[1,2],[9,0],[9,8],[2,3],[8,1]]
y = [0,1,1,2,2,0]

clf = OneVsRestClassifier(estimator=SVC(random_state=0))
print clf.fit(X,y).predict(X)
'''
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit([[0,0],[1,9],[8,7],[2,3],[5,8]],[10,11,12,90,13])
print reg.predict([0,1])
print reg.coef_
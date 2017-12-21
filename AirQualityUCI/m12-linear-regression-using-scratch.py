from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
import random

style.use('fivethirtyeight')
#xs = np.array([1,2,3,4,5,6],dtype=np.float64)
#ys = np.array([5,4,6,5,6,7],dtype=np.float64)
def create_dateset(hm,variance,step=2,correlation=False):
	var = 1
	ys = []
	for i in range(hm):
		y = var+random.randrange(-variance,variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			var += step
		elif correlation and correlation == 'neg':
			var -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope(xs,ys):
	m = ((mean(xs)*mean(ys))-(mean(xs*ys)))/((mean(xs)*mean(xs))-(mean(xs*xs)))
	b = mean(ys)-m*mean(xs)
	return m,b

def squared_error(ys_orig,ys_line):
	return sum((ys_line-ys_orig)**2)
def cod(ys_orig,ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_reg = squared_error(ys_orig,ys_line)
	mean_error = squared_error(ys_orig,y_mean_line)
	return 1-(squared_error_reg/mean_error)

xs,ys = create_dateset(40,100,2,'pos')
m,b = best_fit_slope(xs,ys)
print m,b

regression_line = [(m*x+b) for x in xs]
r_squared = cod(ys,regression_line)
print r_squared
plt.scatter(xs,ys)
plt.plot(xs,regression_line)
new_x=10
plt.scatter(new_x,m*new_x+b,s=200,color='r')
plt.show()
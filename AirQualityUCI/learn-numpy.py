import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(1,11,dtype=float)
y = 2 * x + 5
plt.title('Graph')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.scatter(x,y)
plt.show()

print 
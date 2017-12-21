import numpy as np 
import pandas as pd 
import math
from collections import Counter 
import random

def knn(data,predict,k=3):
	distance = []
	for group in data:
		for features in data[group]:
			euclidian_dist = np.linalg.norm(np.array(features)-np.array(predict))
			distance.append([euclidian_dist,group])

	votes = [i[1] for i in sorted(distance)[:k]]
	#print Counter(votes).most_common(1)
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1]/float(k)
	return vote_result,confidence


df = pd.read_csv('testout.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.4
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

totle = correct = 0
for group in test_set:
	for data in test_set[group]:
		vote,confidence = knn(train_set,data,k=5)
		if vote == group:
			correct += 1
		totle += 1

print ("Accuracy: ",float(correct)/totle)
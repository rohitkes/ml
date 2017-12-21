import pandas as pd
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]

x = list(zip(names,births))
 
df = pd.DataFrame(data=x,columns=['a','b'])

df.to_csv('file.csv',index=False,header=True)
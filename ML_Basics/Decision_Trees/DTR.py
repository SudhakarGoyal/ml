
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
print(len(dataset))
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values
plt.plot(X,Y,'or')

Y = np.reshape(Y,[np.shape(Y)[0],1])

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X,Y)

plt.scatter(X,Y, color = 'red')
plt.plot(X, dtr.predict(X),color='blue')
plt.plot(6.5, dtr.predict(6.5), 'og')

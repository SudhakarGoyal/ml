
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
print(dataset.head(10))
print(len(dataset))
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,-1].values
plt.plot(X,Y,'or')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
linreg = LinearRegression()
linreg.fit(X_poly, Y)

plt.scatter(X,Y, color = 'red')
plt.plot(X, linreg.predict(poly_reg.fit_transform(X)),color='blue')

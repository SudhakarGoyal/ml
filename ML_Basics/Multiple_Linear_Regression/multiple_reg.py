
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
print(dataset.head(10))
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x  = LabelEncoder()   #change text to numbers 
X[:,3] = labelencoder_x.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3]) #convert nubers into binary
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
X = X[:,1:]#removing first column of X

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)

y_pred = reg.predict(X_test)
         



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")

print(dataset.head(10))
sal = dataset["Salary"].values            #Y
exp = dataset["YearsExperience"].values   #X

sal = np.reshape(sal,[len(sal),1])
exp = np.reshape(exp, [len(exp),1])
# =============================================================================
# plt.plot(exp,sal,'or')
# =============================================================================

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(exp, sal, test_size = 1/3, random_state = 0)

#fitting linear reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)

y_hat = lin_reg.predict(X_test)
print(y_hat - Y_test)
sse = np.sum((Y_test - y_hat)**2)
plt.figure()
plt.plot(X_train, Y_train,'or')
plt.plot(X_train, lin_reg.predict(X_train))
plt.xlabel("experience")
plt.ylabel("salary")
plt.legend("linear_reg")
err = (y_hat - Y_test)*100/Y_test
err  = np.mean(err)
print("err is", err)

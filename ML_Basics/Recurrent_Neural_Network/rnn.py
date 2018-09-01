
import numpy as np
from matplotlib import pyplot as plt
import keras
import pandas as pd

#import the training set

data = pd.read_csv("Google_Stock_Price_Train.csv")

training_set = data.iloc[:,1:2].values
print(np.shape(training_set))

#feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
Y_train = []
num_steps = 90

# =============================================================================
# for i in range(0, np.shape(training_set_scaled)[0]):
#     X_train.append(training_set_scaled[i:i+batch_size,0])
#     Y_train.append(training_set_scaled[i+batch_size,0])
# 
# X_train, Y_train = np.array(X_train), np.array(Y_train)
# =============================================================================

for i in range(num_steps, np.shape(training_set_scaled)[0]):
    X_train.append(training_set_scaled[i-num_steps:i,0])
    Y_train.append(training_set_scaled[i,0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1],1))

from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences= True, input_shape = (np.shape(X_train)[1],1))) #first LSTM lYER
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences= True)) #2nd LSMTm layer
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences= True)) #3rd LSMTm layer
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50)) #fourth LSTM lYER
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units = 1, activation = "sigmoid"))


#compile
regressor.compile(optimizer='adam', loss ='mean_squared_error' )

#Fitting the RNN to the training set

regressor.fit(x = X_train, y = Y_train, batch_size= 64, epochs= 100, verbose=1  )


testing_dataset =  pd.read_csv("/home/engineer/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN/Google_Stock_Price_Test.csv")

testing_set = testing_dataset.iloc[:,1:2].values

# =============================================================================
# print(np.shape(testing_set))
# =============================================================================
dataset_total = pd.concat((data['Open'], testing_dataset['Open']), axis = 0)
inputs = dataset_total[(np.shape(dataset_total)[0] - np.shape(testing_set)[0] - num_steps):].values

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(num_steps, np.shape(inputs)[0]):
    X_test.append(inputs[i - num_steps:i,0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_price = regressor.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

plt.plot(testing_set, color = 'red', label = 'Real Stock Prices' )
plt.plot(predicted_price, color = 'green', label = 'Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

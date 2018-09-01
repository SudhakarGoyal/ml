#!/usr/bin/env python

import numpy as np
import math

x = np.array([[0,0],[0,1],[1,0], [1,1]])
y = np.array([[0],[0],[0],[1]])
def sigmoid(x):
    return 1/(1+np.exp(-x))    # defining sigmoid function   -- used as activation function in the hidden layer
def der_sigmoid(x):
    return x*(1-x)        # derivative of sigmoid function 

epoch = 10000
lr = 0.1
input_layer_neurons = x.shape[1]
hidden_layer_neurons = 3
output_layer_neurons = 1
wh = np.zeros((input_layer_neurons,hidden_layer_neurons))
bh = np.random.uniform(size = (1,hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bout = np.random.uniform(size = (1, output_layer_neurons))

#FORWARD PROPAGATION
for i in range(5000):
    hidden_layer_input1 = np.dot(x,wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hidden_layer_activation,wout) 
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    #BACKWARD PROPAGATION
    E = y - output
#     print(E)
    gradient_output_layer = der_sigmoid(output)
    gradient_hidden_layer = der_sigmoid(hidden_layer_activation)
    delta_output = E*gradient_output_layer
    error_hidden_layer =  delta_output.dot(wout.T)
    delta_hidden_layer = error_hidden_layer*gradient_hidden_layer  
    wout+= np.dot(hidden_layer_activation.transpose(),delta_output)*lr    
    bout+= np.sum(delta_output,axis=0,keepdims=True)*lr
    wh+=np.dot(x.transpose(),delta_hidden_layer)*lr
    bh+= np.sum(delta_hidden_layer,axis=0,keepdims=True)*lr
    
print(output)    


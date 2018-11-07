#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:40:26 2018

@author: Sudhakar
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import nltk


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(np.shape(mnist.train.labels)[1])
#nodes in hidden layer 1
n_nodes_hl1 =  64
#nodes in hidden layer 2
n_nodes_hl2 = 128
#nodes in hidden layer 3
n_nodes_hl3 = 128

n_classes = 10
x = tf.placeholder('float', [None, 784]) # dont know number of samples but no of features are 28*28
y = tf.placeholder('float')

def neural_network(data):
    #input_data*weights+ bias
    hidden_layer_1 = {"weights": tf.Variable(tf.truncated_normal([784, n_nodes_hl1])), 
                      "bias":tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #input_data_to layer 2 * weights_layer2 + bias2
    hidden_layer_2 = {"weights": tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2])),
                      "bias": tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer_3 = {"weights": tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3])), 
                      "bias": tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer =   {"weights": tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes])), 
                      "bias": tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_layer_1["weights"]), hidden_layer_1["bias"])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_layer_2["weights"]), hidden_layer_2["bias"])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_layer_3["weights"]), hidden_layer_3["bias"])    
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3,output_layer["weights"]), output_layer["bias"])
    #output = tf.nn.softmax(output)  
    return output

def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    epochs = 10
    num_batches = 32
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/num_batches)):
                _x,_y = mnist.train.next_batch(num_batches)
                _, c = sess.run([optimizer, cost], feed_dict = {x:_x, y:_y})
                epoch_loss+=c
            print("Epoch", epoch, "out of ", epochs, "loss", epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print("accuracy", accuracy.eval({x:mnist.test.images ,y:mnist.test.labels}))
        
        
train_neural_network(x)
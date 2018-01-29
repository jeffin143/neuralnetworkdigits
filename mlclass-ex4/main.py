#!/usr/bin/env python

# 
# Neural network learning
# 
# depends on 
#
#     displayData.py
#     sigmoidGradient.py
#     randInitializeWeights.py
#     nnCostFunction.py
#

import scipy.io
import random
import time
import numpy as np
import displayData as dd
import nnCostFunction as nncf
import sigmoidGradient as sg
import randInitializeWeights as riw
import checkNNGradients as cnng
from scipy.optimize import minimize
import predict as pr

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('dataset.mat')

X = mat["X"]
y = mat["y"]

m = X.shape[0]
print m

# crucial step in getting good performance!
# changes the dimension from (m,1) to (m,)
# otherwise the minimization isn't very effective...
y=y.flatten() 

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100],:]

dd.displayData(sel)

print('Initializing Neural Network Parameters...')

initial_Theta1 = riw.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = riw.randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'), initial_Theta2.reshape(initial_Theta2.size, order='F')))
print('Training Neural Network...')
maxiter=20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
results = minimize(nncf.nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                 (hidden_layer_size, input_layer_size + 1), order='F')

Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                 (num_labels, hidden_layer_size + 1), order='F')

print('\nVisualizing Neural Network... \n')

dd.displayData(Theta1[:, 1:])

raw_input('Program paused. Press enter to continue.\n')

pred = pr.predict(Theta1, Theta2, X)
#print(y)

# code below to see the predictions that don't match
"""
fmt = '{}   {}'
print(fmt.format('y', 'pred'))
for y_elem, pred_elem in zip(y, pred):
	if y_elem == pred_elem:
		#print(fmt.format(y_elem%10, pred_elem%10))
		print(y[y_elem])
		print(pred[pred_elem])
		#raw_input('Program paused. Press enter to continue.\n')
"""
fmt = '{}   {}'
print(fmt.format('y', 'pred'))
for x in range(10):
  p=np.random.randint(1,m)
  pred = pr.predict(Theta1, Theta2, X[p])
  print(fmt.format(y[p]%10, pred%10))
  dd.displayData(X[p])
  time.sleep(.500)


pred = pr.predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:f}'.format( ( np.mean(pred == y)*100 ) ) )




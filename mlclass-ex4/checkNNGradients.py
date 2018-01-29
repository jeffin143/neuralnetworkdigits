import numpy as np
import debugInitializeWeights as diw
import nnCostFunction as nncf
import computeNumericalGradient as cng
from decimal import Decimal

def checkNNGradients(lambda_reg=0):


    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = diw.debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = diw.debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X  = diw.debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(range(m), num_labels).T

    # Unroll parameters
    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

    # Short hand for cost function
    def costFunc(p):
        return nncf.nnCostFunction(p, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lambda_reg)

    _, grad = costFunc(nn_params)
    numgrad = cng.computeNumericalGradient(costFunc, nn_params)

    
    fmt = '{:<25}{}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

  
    
    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))



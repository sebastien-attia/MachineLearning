import numpy as np

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def derivative_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def least_square_error(y, output):
    diff = output - y
    return np.dot(np.transpose(diff), diff)/2

def least_square_error_gradient(y, output):
    return (output - y)

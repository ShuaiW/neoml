# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:34:29 2015

@author: shuaiwang
"""

from __future__ import division
import matplotlib.pylab as plt
import numpy as np
import numpy.linalg as la
from Helper import *


def plot(X, y, theta):
    """
    Given input data X, and their label y, chooses
    two features to plot the data point by different class
    Also plot the decision boundary using theta.
    
    :Parameters:
        X: training input data
        y: training labels
        theta: parameters
        
    :Returns:
        A plot of binary-class data points and the decision boundary.
    """
    y = y.reshape(y.size)
    theta = theta.reshape(theta.size)
    one = X[y==1, 1:]
    zero = X[y==0, 1:]
    plt.scatter(one[:, 0], one[:, 1], c = 'g', marker='x')
    plt.scatter(zero[:, 0], zero[:, 1])
    
    xx = np.array([min(X[:, 1]), max(X[:, 1])])
    yy = -theta[0] / theta[2] - theta[1] / theta[2] * xx
    plt.plot(xx, yy, '--', c='r')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('Binary-class and the decision boundary')
    plt.show()    


def loss(X, theta, y):
    """
    Computers logistic log loss.
    
    :Parameters:
        X: training input data
        theta: parameters
        y: training labels
        
    :Returns:
        A number, the loss.
    """
    m = len(y)
    hx = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(hx) + (1-y) * np.log(1-hx)) / m
    

def sigmoid(X):
    """
    Vectorized sigmoid function.
    
    :Parameters:
        X: input value
        
    :Returns: 
        sigmoidal value
    """
    return 1.0 / (1.0 + np.exp(-X))
    

def descent_logistic(X, y, alpha=0.003):
    """
    Logistic Regression.
    Divided update by # of examples -- otherwise need extremly small alpha.

    :Parameters:
        X: training input data
        y: training outputs
        alpha: learning rate

    :Returns: 
        a parameter vector
    """
    X = add_bias(X)
    theta = np.zeros( (X.shape[1], 1) )
    m = len(y)
    
    for i in range(2000):
        hx = sigmoid(X.dot(theta))
        theta -= alpha * X.T.dot(hx - y) / m
    return theta

  
def newton_logistic(X, y, max_iter=3000):
    """
    Applies Newton's method to optimize the log likelihood of logistic 
    regression, using batch method.
    
    :Parameters:
        X: training input data
        y: training output data

    :Returns: 
        theta
    """
    X = add_bias(X)
    theta = np.zeros( (X.shape[1], 1) )
    
    for i in range(max_iter):
        hx = sigmoid(X.dot(theta))
        hessian = -hx.T.dot((1-hx)) * X.T.dot(X)
        theta -= la.inv(hessian).dot(X.T.dot((y-hx)))

    plot(X, y, theta)
    return theta


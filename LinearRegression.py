# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:57:24 2015

@author: shuaiwang
"""

from __future__ import division
import numpy as np
import numpy.linalg as la
from Helper import *


def plot(X, y, theta):
    """
    Given input data X, and their value y, scatter plot them
    as well as plot the best fit line.
    
    :Parameters:
        X: training input data
        y: training labels
        theta: parameters
        
    :Returns:
        A scatter plot with best fit line
    """
    plt.scatter(X, y)
    xx = np.linspace(min(X), max(X), 100)
    yy = theta[1][0] * xx + theta[0][0]
    plt.plot(xx, yy, c='r')
    plt.title('Scatter plot and best fit line')
    plt.show()    


def loss(X, y, theta):
    """
    Computers mean squared loss.
    
    :Parameters:
        X: training input data
        y: training outputs
        theta: parameters
        
    :Returns:
        A number, the loss.
    """
    print 'The loss is {0}.'.format(0.5 * np.sum((X.dot(theta) - y)**2))
    

def gradient_descent(X, y, alpha=0.01, num_iter=1500):
    """
    Gradient descent for linear regression. Batch update. If dataset is large,
    consider stochastic method.
    Divided update by # of examples -- otherwise need extremly small alpha.
    
    :Parameters:
        X: training input data
        y: training outputs
        alpha: learning rate
        num_iter: number of iterations
        
    :Returns: 
        theta and loss
    """
    X = add_bias(X)
    theta = np.zeros( (X.shape[1], 1) ) 
    m = len(y)
    J_history = np.zeros(num_iter)

    for i in range(num_iter):
        theta -= alpha * X.T.dot((X.dot(theta) - y)) / m
        J_history[i] = loss(X, theta, y)
    return theta, J_history


def stochastic_gradient_descent(X, y, alpha=0.01, num_iter=150):
    """
    Stochastic gradient descent for linear regression. Update parameters
    using one training example at a time.
    Divided update by # of examples -- otherwise need extremly small alpha.
    
    :Parameters:
        X: training input data
        y: training outputs
        alpha: learning rate
        num_iter: number of iterations

    :Returns: 
        theta and loss
    """
    X = add_bias(X)
    theta = np.zeros( (X.shape[1], 1) ) 
    m = len(y)
    J_history = np.zeros(num_iter * m)
    
    for i in range(num_iter):
        for j in range(m):
            theta -= alpha * X[[j]].T.dot((X[[j]].dot(theta) - y[[j]])) / m
        J_history[i] = loss(X, theta, y)
    return theta, J_history


def normal_equation(X, y):
    """
    Closed form solution for linear regression.
    
    :Parameters:
        X: training input data
        y: training outputs
        
    :Returns: 
        a parameter vector
    """
    X = add_bias(X)
    theta = la.pinv(X.T.dot(X)).dot(X.T).dot(y)
    
    plot(X[:, 1:], y, theta)
    loss(X, y, theta)
    
    return theta
    

def locally_weighted_normal_equation(X, y):
    """
    A non-parametric regression method using normal equation.
    
    :Parameters:
        X: training input data
        y: training outputs
        tau: bandwidth

    :Returns: 
        a plot with different values of tau.
    """
    taus = [.1, .3, 1, 3, 10]
    colors = ['r', 'g', 'b', 'y', 'k']
    M = X.shape[0]
    query_x = np.linspace(min(X), max(X), M)
    query_y = np.zeros(len(query_x))
    X = add_bias(X)
    plt.figure(figsize=(15, 10))
    
    for i in range(len(taus)):
        tau = taus[i]
        for k in range(len(query_x)):
            W = np.zeros((M, M))
            for j in range(M):
                W[j, j] = np.exp(-(query_x[k] - X[j, 1])**2 / (2 * tau**2))
            theta = la.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
            query_y[k] = theta[0][0] + theta[1][0] * query_x[k]
        plt.plot(query_x, query_y, c=colors[i], label='tau={0}'.format(tau))
    
    plt.scatter(X[:, 1], y)
    plt.title('Locally Weight Regression with Different Values of Tau')
    plt.xlim(-6, 13)
    plt.ylim(-2, 2.8)
    plt.legend(loc='lower right')
    plt.show()


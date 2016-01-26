# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:34:47 2015

@author: shuaiwang
"""

from __future__ import division
import numpy as np
from random import randrange

def add_bias(X):
    """
    Takes a matrix of size (M, N), and adds a column of 1's (bias) on the left.
    
    :Param:
        X: design matrix of size (M, N)
        
    :Returns: 
        New matrix with bias column, of size (M, N+1)
    
    [2, 3, 4, 5]    =>         [1, 2, 3, 4, 5]
    [6, 7, 8, 9]               [1, 6, 7, 8, 9]
    """
    M = X.shape[0]
    return np.c_[ np.ones(M), X ]

    
def softmax(X):
    """
    A softmax classifier (usually as the final layer of NN)
    
    :Param:
        X: a (M, K) dimension matrix where M is the number of examples,
            and K is the number of classes. 
    
    :Return:
        a matrix of same size as X, where the (M, K)th entry is the 
        probabilty that example M is classified under class K.
    """
    try:
        return np.exp(X) / np.sum(np.exp(X), axis=1)[:, None]
    except:    # deal with vector
        return np.exp(X) / np.sum(np.exp(X))
        

def grad_check(f, w, num_checks=10, h=1e-5):
    """
    Samples a few random elements and checks if numerical and 
    analytical gradient match.
    
    :Param:
        f: a lambda function that takes w returns tuple (loss, dW).
        W: weights of size (N, K) N-> dimension; K-> number of classes
    
    :Return:
        a matrix of same size as X, where the (M, K)th entry is the 
        probabilty that example M is classified under class K.
    """
    analytic_grad = f(w)[1]         # analytical gradient
    
    for i in xrange(num_checks):
        ix = tuple([randrange(m) for m in w.shape])    # (row, col)
        
        oldval = w[ix]
        w[ix] = oldval + h          # increment by h
        fxph = f(w)[0]              # evaluate f(w + h)
        w[ix] = oldval - h          # increment by h
        fxmh = f(w)[0]              # evaluate f(w - h)
        w[ix] = oldval              # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)

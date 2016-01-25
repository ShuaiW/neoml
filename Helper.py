# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:34:47 2015

@author: shuaiwang
"""

from __future__ import division
import numpy as np


def add_bias(X):
    """
    Takes a matrix of size (M, N), and adds a column of 1's (bias) on the left.
    
    :Parameters:
        X: design matrix of size (M, N)
        
    :Returns: 
        New matrix with bias column, of size (M, N+1)
    
    [2, 3, 4, 5]    =>         [1, 2, 3, 4, 5]
    [6, 7, 8, 9]               [1, 6, 7, 8, 9]
    """
    M = X.shape[0]
    return np.c_[ np.ones(M), X ]


def regresssion_predict(X, W):
    """
    Given some data (without bias), X, and a pre-trained parameter vector, W, 
    predicts the outcome.
    
    :Parameters:
        X: (M, N) data we want make prediction on
        W: (N+1,) a pre-trained set of parameters
        
    :Returns: 
        prediction for the data (M, )
    """
    return add_bias(X).dot(W)

    
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
        


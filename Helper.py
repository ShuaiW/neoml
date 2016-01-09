# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:34:47 2015

@author: shuaiwang
"""

from __future__ import division
import numpy as np


def add_bias(X):
    """
    Takes a matrix of size m x n, and adds a column of 1's (bias) on the left.
    
    :Parameters:
        X: matrix, m x n
        
    :Returns: 
        New matrix with bias column, of size m x (n+1)
    
    [2, 3, 4, 5]    =>         [1, 2, 3, 4, 5]
    [6, 7, 8, 9]               [1, 6, 7, 8, 9]
    """
    m = len(X)
    return np.c_[ np.ones(m), X ]


def predict(X_unseen, theta):
    """
    Given a trained parameter vector and some unseen data, predicts
    the outcome.
    
    :Parameters:
        X_unseen: unseen data we want make prediction on
        theta: a trained set of parameters
        
    :Returns: 
        prediction for the unseen data
    """
    return add_bias(X_unseen).dot(theta)

    
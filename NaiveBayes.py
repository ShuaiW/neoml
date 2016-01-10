# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:47:02 2016

@author: shuaiwang
"""

from __future__ import division
import numpy as np


class NaiveBayes:
    """
    A multinomial Naive Bayes spam classifier.
    """
    
    def __init__(self, train_X, train_y, test_X, test_y):
        """
        :Parameters:
            train_X: training input data
            train_y: training labels
            test_X: testing input data.
            test_y: testing labels
        
        train_X, test_X: matrix where the (i, j)-entry represents the number 
            of occurences of the jth token in the ith documents.
        """
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y


    def train(self):
        """
        Trains a NB spam classifier.
        """
        # term to add in denominator in lapalce smoothing
        V = self.train_X.shape[1]
        
        pos = self.train_X[self.train_y == 1, :]
        neg = self.train_X[self.train_y == 0, :]
        
        # total words in both classes
        pos_words = sum(sum(pos))
        neg_words = sum(sum(neg))
        
        # prior
        self.pos_log_prior = np.log(len(pos) / len(self.train_X))
        self.neg_log_prior = np.log(len(neg) / len(self.train_X))
        
        # log likelihood with laplace smoothing
        self.pos_log_phi = np.log((sum(pos)+1) / (pos_words + V))
        self.neg_log_phi = np.log((sum(neg)+1) / (neg_words + V))
        
    
    def test(self):
        """
        Tests a NB spam classifier.
        """
        # posterior
        pos_posterior = self.test_X.dot(self.pos_log_phi.T) + self.pos_log_prior
        neg_posterior = self.test_X.dot(self.neg_log_phi.T) + self.neg_log_prior
        
        prediction = pos_posterior > neg_posterior
        
        print 'Test error is {0}'.format(np.mean(prediction != self.test_y))

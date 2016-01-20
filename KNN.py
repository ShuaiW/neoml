# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:33:13 2016

@author: shuaiwang
"""
import numpy as np


class KNearestNeighbor:
    """ a kNN classifier with L2 distance """
  
    def __init__(self):
        pass


    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.
    
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
            consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
            y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.
    
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data 
            consisting of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
    
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for
            the test data, where y[i] is the predicted label for the test 
            point X[i].  
        """
        dists = self.compute_distances(X)

        return self.predict_labels(dists, k=k)


    def compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training 
        point in self.X_train.

        Inputs:
            - X: A numpy array of shape (num_test, D) containing test data.

        Returns:    
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            is the Euclidean distance between the ith test point and the jth 
            training point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
    
        # broadcast and vectorization; (x-y)**2 = x**2 - 2xy + y**2
        test_sqr = np.sum(X**2, axis=1).reshape(num_test, 1)
        train_sqr = np.sum(self.X_train**2, axis=1).reshape(1, num_train)
        cross = X.dot(self.X_train.T)
        dists = test_sqr - 2 * cross + train_sqr

        return dists


    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
    
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
              gives the distance betwen the ith test point and the jth training 
              point.
    
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for 
            the test data, where y[i] is the predicted label for the test 
            point X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # find the indices of the most similar items in the training set
            idxs = np.argsort(dists[i, :])[:k]
            # get their training labels
            closest_y = self.y_train[idxs]
            # do a majority vote
            y_pred[i] = np.argmax(np.bincount(closest_y))
            
        return y_pred
    

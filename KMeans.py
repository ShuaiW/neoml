# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:51:16 2015

@author: shuaiwang
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm


def plot(C, X, indices, num_iter):
    """
    Visualization for k means clustering. Works only for 2-D.
    
    :Parameters:
        C: centroids
        X: training input data
        indices: centroid indices for X
        num_iter: the number of iteration we are at
    
    :Returns: 
        K means plot
    """
    # Margin setup
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    x_margin, y_margin = 0.2 * (x_max - x_min), 0.2 * (y_max - y_min)
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    K = len(C)
    colors = cm.rainbow(np.linspace(0.1, 1, K))
    
    # plot unclustered data
    if num_iter == 0:
        plt.title("Unclustered Data")
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        
    for j in range(K):
        # before the iteration, plot the initialized centroids
        if num_iter == 0:
            plt.title("Initial Centroids")
            plt.scatter(C[j][0], C[j][1], s=100, c=colors[j], marker='x')
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)
        # after each iteration, plot centroids and belonging points
        else:
            plt.title("Results after {0}'s iteration".format(num_iter))
            plt.scatter(C[j][0], C[j][1], s=100, c=colors[j], marker='x')
            plt.scatter(X[indices==j][:, 0], X[indices==j][:, 1], c=colors[j])
    # show plot after this iteration
    plt.show()
    

def distortion_func(X, C):
    """
    Measrues the sum of squared distance between each training example
    and the centroid to which it has been assigned.
    
    :Parameters:
        X: training input data
        C: centroids
        
    :Returns:
        A number, called "distortion" (like loss).
    """
    return sum(min(np.sum((x-C)**2, axis=1)) for x in X)

    

def kmeans(X, K):
    """
    K means clustering algorithm.
    Convergence condition: no change in assignment.
    
    :Parameters:
        X: training input data
        K: number of centroids
        
    :Returns:
        centroids loc and distortion history.
    """
    # dimension of the training data
    M, N = X.shape
    
    # distortion list for debug
    distortion_history = []
    
    # initialize cluster centroids randomly
    centroids = np.random.uniform(X.min(axis=0), X.max(axis=0), [K, N])
    
    # initialize an random array
    last_assign = np.repeat(K+1, M)
    
    # plot initial centroids and unclustered data
    plot(centroids, X, last_assign, 0)

    # num of iterations    
    num_iter = 1   
    
    while True:  
        # a container to hold centroid indice for each data point
        new_assign = []
        
        # assign to the closest contriod
        for x in X:
            new_assign.append(np.argmin(np.sum((x - centroids)**2, axis=1)))
        
         # check convergence
        if (last_assign == new_assign).all(): 
            print "\n\nThe algorithm converged at the {0}'s iterations.".format(num_iter)
            print "Convergence means no reassignment of centroid for all data points at this iteration."
            break

        # convert to array for vectorized operation
        last_assign = new_assign = np.array(new_assign)
        
        # get distortion (loss) BEFORE recompute the centroid locations
        distortion_history.append(distortion_func(X, centroids))
        
        # recompute centroid postiion
        for j in range(K):
            centroids[j] = np.mean(X[new_assign == j], axis=0)
        
        ### after each iteration
        # plot the new centroids along with data belong to this
        plot(centroids, X, new_assign, num_iter)
        # increment
        num_iter += 1
        
    return centroids, distortion_history
    
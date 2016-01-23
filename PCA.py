# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:15:26 2015

@author: shuaiwang
"""

from __future__ import division
from PIL import Image
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def represent(img_file, k):
    """
    Reresents a grayscale image using fewer components.
    
    :Parameters:
        img_file: an image file.
        k: number of components.
        
    :Returns:
        compressed image
    """
    print '\n'
    
    img = np.asarray(Image.open(img_file))
    print 'Original image'
    plt.imshow(img)
    plt.show()
    
    print '\n'
    
    # Using first k component
    reconstruct = pca(img, k)
    r = k / img.shape[1] * 100    # principle component ratio
    print 'Represent image using the first {0}% principle components'.format(r)
    plt.imshow(reconstruct)
    plt.show()
    
    
def pca(X, k):
    """
    Principle component analysis. 
    
    :Parameters:
        X: training input data
        k: number of principle components
        
    :Returns:
        Reconstruct matrix using first k eigenvectors.
    """
    U, S, VT = la.svd(X)
    K = U[:, :k]
    return K.dot(K.T).dot(X)


represent('flower.jpg', 25)
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
    Reresents the img in lower dimensions. 
    
    :Parameters:
        img: an image file.
        k: number of dimensions.
        
    :Returns:
        compressed image
    """
    img= Image.open(img_file)
    print 'Original image'
    plt.imshow(img)
    plt.show()
    
    print '\n'
    
    # Using first k component
    data = np.asarray(img)
    reconstruct = pca(data, k)
    print 'Image using the first {0} components'.format(k)
    plt.imshow(reconstruct)
    
    
def pca(X, k):
    """
    Principle component analysis. 
    
    :Parameters:
        X: training input data
        k: number of principle components
        
    :Returns:
        Reconstruct matrix using first k eigenvectors.
    """
    U, D, V_T = la.svd(X)
    R = V_T.T[:, :k]
    return R.dot(R.T).dot(X)
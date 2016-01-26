# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 22:02:26 2016

@author: shuaiwang
"""
import numpy as np
from Helper import grad_check


def svm_loss(W, X, y, reg):
    """
    Structured SVM loss function (hinge loss).

    Params:
        W: A numpy array of shape (D, C) containing weights.
        X: A numpy array of shape (N, D) containing a minibatch of data.
        y: A numpy array of shape (N,) containing training labels; y[i] = c 
            means that X[i] has label c, where 0 <= c < C.
        reg: (float) regularization strength

    Returns a tuple of:
        loss as single float
        gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)      # initialize the gradient as zero
    delta = 1                   # 'margin'
    num_train = X.shape[0]
    
    # compute loss
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y]
    margin = scores - correct_class_scores[:, None]
    margin[margin != 0] += delta  # add delta except for the scores from correct class
    loss = np.maximum(0, margin)
    loss = np.sum(loss) / num_train + 0.5 * reg * np.sum(W * W)  # 1/N and add bias
    
    # compute gradient
    sj = margin > 0        # mask of scores from other classes
    syi = margin == 0      # mask of score of correct class
    num_over_zero = np.sum(margin > 0, axis=1)
    num_syi = syi * num_over_zero[:, None]
    dW += X.T.dot(sj)
    dW -= X.T.dot(num_syi)
    dW = dW / num_train + reg * W           # 1/N and add bias

    return loss, dW


def softmax_loss(W, X, y, reg):
    """
    Softmax loss function using cross entropy.

    Params:
        W: A numpy array of shape (D, C) containing weights.
        X: A numpy array of shape (N, D) containing a minibatch of data.
        y: A numpy array of shape (N,) containing training labels; y[i] = c 
            means that X[i] has label c, where 0 <= c < C.
        reg: (float) regularization strength

    Returns a tuple of:
        loss as single float
        gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    
    # compute loss
    scores = X.dot(W)
    row_max = np.max(scores, axis=1)
    scores -= row_max[:, None]   # 'shift' to guarantee numeric stability
    correct_class_scores = scores[np.arange(num_train), y]
    row_sum_exp = np.sum(np.exp(scores), axis=1)
    softmax = np.exp(correct_class_scores) / row_sum_exp
    loss = np.sum(-np.log(softmax)) / num_train + 0.5 * reg * np.sum(W * W)

    # compute gradient
    all_softmax = np.exp(scores) / row_sum_exp[:, None]
    all_softmax[np.arange(num_train), y] -= 1    # -1 for prob of correct class
    dW = X.T.dot(all_softmax) / num_train + reg * W
    
    return loss, dW


# gradient check
svm = lambda w: svm_loss(W, X, y, 1e2)
grad_check(svm, W)

softmax = lambda w: softmax_loss(W, X, y, 1e2)
grad_check(softmax, W)

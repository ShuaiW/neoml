import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    c, h, w = input_dim
    
    self.params['W1'] = weight_scale * np.random.randn(num_filters, c, 
        filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(h/2 * w/2 * num_filters,
        hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    N = X.shape[0] 
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # following setting makes sure the output and X are of the same 
    # height and width after conv, and the result after max poll will be
    # height/2, and width/2
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    # >>>>>>>>>>>>>>>>>>>> forward pass >>>>>>>>>>>>>>>>>>>>
    out_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out_2, cache_2 = affine_relu_forward(out_1, W2, b2)
    scores, cache_3 = affine_forward(out_2, W3, b3)
    
    if y is None:
      return scores
      
    # loss
    loss, _ = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**3))
    
    # <<<<<<<<<<<<<<<<<<<<< back prop <<<<<<<<<<<<<<<<<<<<<
    dout = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    dout /= np.sum(dout, axis=1, keepdims=True)
    dout[np.arange(N), y] -= 1
    dout /= N
    
    dout_2, dW3, db3 = affine_backward(dout, cache_3)
    dout_1, dW2, db2 = affine_relu_backward(dout_2, cache_2)
    _, dW1, db1 = conv_relu_pool_backward(dout_1, cache_1)
        
    # L2 reg
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    
    grads = {}
    
    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3
    
    
    return loss, grads
  
  

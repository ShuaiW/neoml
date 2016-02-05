import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    # unpack hyperparameters
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N = X.shape[0]
    
    # >>>>>>>>>>>>>>>>>>>> forward pass >>>>>>>>>>>>>>>>>>>>
    first_out, first_cache = affine_relu_forward(X, W1, b1)
    scores, cache = affine_forward(first_out, W2, b2)

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    # loss 
    loss, _ = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
    
    #  <<<<<<<<<<<<<<<<<<<<< back prop <<<<<<<<<<<<<<<<<<<<<
    dout = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    dout /= np.sum(dout, axis=1, keepdims=True)
    dout[np.arange(N), y] -= 1
    dout /= N
    
    dfirst_out, dW2, db2 = affine_backward(dout, cache)
    dX, dW1, db1 = affine_relu_backward(dfirst_out, first_cache)
    
    # L2 reg
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    
    grads = {}
    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    
    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    for i in xrange(self.num_layers):
      W, b = 'W' + str(i+1), 'b' + str(i+1)
      if i == 0:                      # first hidden layer
        prev = input_dim
        cur = hidden_dims[i]
      elif i == self.num_layers-1:    # last fc layer
        prev = hidden_dims[-1]
        cur = num_classes
      else:                           # middle hidden layers (if exist)
        prev = hidden_dims[i-1]
        cur = hidden_dims[i]  
     
      self.params[W] = weight_scale * np.random.randn(prev, cur)
      self.params[b] = np.zeros(cur)
      
      # if batchnorm is used, initialize parameters
      if self.use_batchnorm and i < self.num_layers-1:
         gamma, beta = 'gamma' + str(i+1), 'beta' + str(i+1)
         self.params[gamma] = np.ones(cur)
         self.params[beta] = np.zeros(cur)


    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). Here we use the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. We should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    N = X.shape[0]
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    # >>>>>>>>>>>>>>>>>>>> forward pass >>>>>>>>>>>>>>>>>>>>    
    forward = {}    
    for i in xrange(self.num_layers):
      out, cache = 'out_' + str(i+1), 'cache_' + str(i+1)
      
      # first forward: use input data, X; 
      # following forward: use 'out' from previous layer
      x = X if i == 0 else forward['out_' + str(i)]

      if i == self.num_layers - 1:      # last fc layer
        forward['scores'], forward[cache] = affine_forward(x,
            self.params['W' + str(i+1)], self.params['b' + str(i+1)])
      else:                             # hidden layers
        if self.use_batchnorm:
          forward[out], forward[cache] = affine_batchnorm_relu_forward(x, 
              self.params['W' + str(i+1)], self.params['b' + str(i+1)],
              self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)],
              self.bn_params[i])
        else:
          forward[out], forward[cache] = affine_relu_forward(x, 
              self.params['W' + str(i+1)], self.params['b' + str(i+1)])
        
        # if dropout is used, add it after each nonlinearity
        if self.use_dropout:
          forward[out], forward['dropout_cache' + str(i+1)] = \
              dropout_forward(forward[out], self.dropout_param)
        
    scores = forward['scores']
    
    # If test mode return early
    if mode == 'test':
      return scores

    # loss    
    loss, _ = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(np.sum(self.params[W]**2) for 
        W in self.params if W.startswith('W'))

    
    # <<<<<<<<<<<<<<<<<<<<< back prop <<<<<<<<<<<<<<<<<<<<<
    dscores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    dscores /= np.sum(dscores, axis=1, keepdims=True)
    dscores[np.arange(N), y] -= 1
    dscores /= N
    
    grads, grads_out = {}, {}

    for i in xrange(self.num_layers, 0, -1):
      # first back prop: use dscore as dout
      # following back prop: use 'dout' from previous layer
      dout = dscores if i == self.num_layers else grads_out['dout_' + str(i)]
      
      if i == self.num_layers:      # last fc layer
        grads_out['dout_'+str(i-1)], grads['W'+str(i)], grads['b'+str(i)] = \
            affine_backward(dout, forward['cache_' + str(i)])
      else:                         # other layers
        # accordinly, dropout before nonlinearity if it's used        
        if self.use_dropout:
          dout = dropout_backward(dout, forward['dropout_cache'+str(i)])
        
        if self.use_batchnorm:
          grads_out['dout_'+str(i-1)], grads['W'+str(i)], grads['b'+str(i)], \
          grads['gamma'+str(i)], grads['beta'+str(i)] = \
              affine_batchnorm_relu_backward(dout, forward['cache_' + str(i)])
        else:
          grads_out['dout_'+str(i-1)], grads['W'+str(i)], grads['b'+str(i)] = \
              affine_relu_backward(dout, forward['cache_' + str(i)])
    
    # L2 reg (no reg for scale and shift parameters)
    for i in xrange(1, self.num_layers+1):
        grads['W' + str(i)] += self.reg * self.params['W' + str(i)]

    return loss, grads

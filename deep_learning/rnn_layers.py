import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
  cache = (x, prev_h, Wx, Wh, b, next_h)
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state, (N, H)
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients to previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  x, prev_h, Wx, Wh, b, next_h = cache # unpack cahce
  
  dtanh = (1 - next_h * next_h) * dnext_h # (N, H)
  db = np.sum(dtanh, axis=0) # (H,)
  dWh = prev_h.T.dot(dtanh) # (H, H)
  dprev_h = dtanh.dot(Wh.T) # (N, H)
  dWx = x.T.dot(dtanh) # (D, H)
  dx = dtanh.dot(Wx.T) # (N, D)
  
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  N, T, _ = x.shape; _, H = h0.shape
  h = np.zeros((N, T, H))
  
  for t in range(T):
    if t == 0: prev_h = h0 # initial h; get the machine running
    h[:, t, :], _ = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b) # forward update
    prev_h = h[:, t, :] # in the next iter, cur becomes prev

  cache = (x, h0, Wx, Wh, b, h)
  
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream (layer y) gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  (x, h0, Wx, Wh, b, h) = cache
  N, T, H = dh.shape
  dx, dWx = np.zeros(x.shape), np.zeros(Wx.shape)
  dWh, db = np.zeros(Wh.shape), np.zeros(b.shape)
  
  dhori = np.zeros((N, H))  # gradient contributed by the horizontal branch
  for t in reversed(range(T)):
    prev_h = h0 if t == 0 else h[:, t-1, :]
    _cache = x[:, t, :], prev_h, Wx, Wh, b, h[:, t, :]
    dx[:, t, :], dhori, _dWx, _dWh, _db = rnn_step_backward(dh[:, t, :] + dhori, _cache)
    dWx += _dWx
    dWh += _dWh
    db += _db
  dh0 = dhori # h0 (last one in the chain) has gradident only contributed horizontally
  
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out = W[x]
  cache = (W, x)
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  W, x = cache
  dW = np.zeros(W.shape)
  # np.add.at (array, position, value to add); inspired by native version:
  # double loops N and T, dW[x[n, t]] += dout[n, t, :]
  np.add.at(dW, x, dout) 
  
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  a = x.dot(Wx) + prev_h.dot(Wh) + b                                # 1
  ai, af, ao, ag = np.split(a, 4, axis=1)                           # 2
  i, f, o, g = sigmoid(ai), sigmoid(af), sigmoid(ao), np.tanh(ag)   # 3
  next_c = f * prev_c  + i * g                                      # 4
  tanh_c = np.tanh(next_c)                                          # 5
  next_h = o * tanh_c                                               # 6
  
  cache = (x, Wx, prev_h, Wh, i, f, o, g, prev_c, tanh_c)
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  # unpack cache
  (x, Wx, prev_h, Wh, i, f, o, g, prev_c, tanh_c) = cache
  
  do = tanh_c * dnext_h                   # 6
  dtanh_c = o * dnext_h
  dnext_c += (1 - tanh_c**2) * dtanh_c    # 5
  df = prev_c * dnext_c                   # 4
  dprev_c = f * dnext_c
  di = g * dnext_c
  dg = i * dnext_c
  dai = (1-i)*i * di                      # 3
  daf = (1-f)*f * df
  dao = (1-o)*o * do
  dag = (1-g**2) * dg
  da = np.hstack((dai, daf, dao, dag))    # 2
  dx = da.dot(Wx.T)                       # 1
  dWx = x.T.dot(da)
  dprev_h = da.dot(Wh.T)
  dWh = prev_h.T.dot(da)
  db = np.sum(da, axis=0)
  
  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  N, T, _ = x.shape; H = h0.shape[1]
  h = np.zeros((N, T, H))
  c0 = np.zeros(h0.shape)
  
  # cache for back prop
  c = np.zeros((N, T, H))
  i, f = np.zeros((N, T, H)), np.zeros((N, T, H))
  o, g = np.zeros((N, T, H)), np.zeros((N, T, H))
  tanh_c = np.zeros((N, T, H))
  
  for t in xrange(T):
    if t==0: 
      prev_h, prev_c = h0, c0
    h[:,t,:], c[:,t,:], _cache = lstm_step_forward(x[:,t,:], prev_h, prev_c, Wx, Wh, b)  
    prev_h , prev_c = h[:, t, :], c[:, t, :] # cur used as prev for next iter
    _, _, _, _, i[:,t,:], f[:,t,:], o[:,t,:], g[:,t,:], _, tanh_c[:,t,:] = _cache

  cache = (x, Wx, h0, Wh, b, i, f, o, g, c, tanh_c, h)
  
  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  (x, Wx, h0, Wh, b, i, f, o, g, c, tanh_c, h) = cache # unpack cache
  
  N, T, H = dh.shape
  dx, dWx = np.zeros(x.shape), np.zeros(Wx.shape)
  dWh, db = np.zeros(Wh.shape), np.zeros(b.shape)
  
  dhori = np.zeros((N, H)) # gradient contributed horizontally
  dprev_c = np.zeros((N, H)) # initial dprev_c
  
  for t in reversed(range(T)):
    if t == 0:
      prev_h, prev_c = h0, np.zeros(h0.shape) # initial cell zero
    else:
      prev_h, prev_c = h[:,t-1,:], c[:,t-1,:]

    _cache = (x[:,t,:], Wx, prev_h, Wh, i[:,t,:], f[:,t,:], o[:,t,:],
              g[:,t,:], prev_c, tanh_c[:,t,:])
    _dx, dhori, dprev_c, _dWx, _dWh, _db = lstm_step_backward(dh[:,t,:]+dhori, dprev_c, _cache)
    dx[:,t,:] = _dx
    dWx +=  _dWx
    dWh += _dWh
    db += _db
  dh0 = dhori # h0 (last one in the chain) has gradident only contributed horizontally
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = x.reshape(N * T, D).T.dot(dout.reshape(N * T, M))
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """
  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T) # if zero then no contribution to loss
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True)) # shift for numerical stability
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N # correct class
  
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1 # -1 for prob of correct class
  dx_flat /= N
  dx_flat *= mask_flat[:, None] # kill gradient if no contribution
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx


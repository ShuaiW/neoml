import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  # with momentum update, the parameter vector will build up velocity 
  # in any direction that has consistent gradient.
  mu = config['momentum']
  v = mu * v - config['learning_rate'] * dw     # integrate velocity
  next_w = w + v                                # integrate position
  
  # update velocity
  config['velocity'] = v

  return next_w, config


def rmsprop(w, dw, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(w))

  # update rule from slide 29 of Lecture 6 of Geoff Hinton's Coursera class
  cache = config['decay_rate'] * config['cache'] + (1-config['decay_rate']) * dw**2
  next_w = w + (-config['learning_rate'] * dw) / (np.sqrt(cache) + config['epsilon'])

  # update cache
  config['cache'] = cache
  
  return next_w, config


def adam(w, dw, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(w))
  config.setdefault('v', np.zeros_like(w))
  config.setdefault('t', 0)
  
  # original paper http://arxiv.org/pdf/1412.6980v8.pdf
  t = config['t'] + 1
  m = config['beta1'] * config['m'] + (1-config['beta1']) * dw
  v = config['beta2'] * config['v'] + (1-config['beta2']) * (dw**2)
  
  # use bias_corrected moments to update w
  mb = m / (1 - config['beta1']**t)
  vb = v / (1 - config['beta2']**t)
  next_w = w - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
  
  # update m, v, and t
  config['m'] = m
  config['v'] = v
  config['t'] = t
  
  return next_w, config
import numpy as np
from random import shuffle
#from past.builtins import range

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  eps = 1e-9

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(N):
    # X[i] (D,)
    prod = np.dot(X[i], W) # (C,)
    prod = np.exp(prod - np.max(prod)) # Numerical stability
    div = prod / (np.sum(prod) + eps)
    loss += - np.log(div[y[i]])
    div[y[i]] -= 1
    dW += np.dot(X[i][:, np.newaxis], div[np.newaxis, :])

  loss /= N
  dW /= N
  loss += 0.5 * reg * np.sum(W**2) 
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  eps = 1e-9
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  prod = np.dot(X, W) # (N,C)
  prod = np.exp(prod - np.max(prod, axis=1, keepdims=True))
  div = prod / (np.sum(prod, axis=1, keepdims=True) + eps) # (N,C)
  loss = - np.sum(np.log(div[range(N), y]))
  div[range(N), y] -= 1
  dW = np.dot(X.T, div)

  loss /= N
  dW /= N
  loss += 0.5 * reg * np.sum(W**2) 
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


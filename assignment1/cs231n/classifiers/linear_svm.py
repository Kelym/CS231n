import numpy as np
from random import shuffle
#from past.builtins import range

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W**2)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    in_loss = scores - correct_class_score + 1 > 0
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] -= X[i] * (np.sum(in_loss) - 1) 
      elif in_loss[j]:
        dW[:, j] += X[i] 
  dW /= num_train
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  scores = np.dot(X, W) # (N,C)
  correct_class_score = scores[range(N), y] # (N)
  loss_matrix = scores - correct_class_score[:, np.newaxis] + 1 #(N,C)
  loss_matrix[range(N), y] = 0
  in_loss = loss_matrix > 0 # (N,C)
  loss = np.sum(loss_matrix * in_loss)
  loss /= num_train
  loss += reg * np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  bool_loss = np.zeros(in_loss.shape)
  bool_loss[in_loss] = 1
  bool_loss[range(N), y] = -np.sum(bool_loss, axis=1)
  dW = np.dot(X.T, bool_loss)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

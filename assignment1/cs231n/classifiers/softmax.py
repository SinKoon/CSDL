import numpy as np
#from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  N, C = X.shape[0], W.shape[1]
  for i in range(N):
      val = np.dot(X[i], W)
      val -= np.max(val) # val.shape = C
      loss = loss + np.log(np.sum(np.exp(val))) - val[y[i]]
      dW[:, y[i]] -= X[i]
      for j in range(C):
          dW[:, j] += np.exp(val[j]) / np.exp(val).sum() * X[i]
  loss = loss / N + 0.5 * reg * np.sum(W * W)        #regulization
  dW = dW / N + reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  N = X.shape[0]
  val = np.dot(X, W) # val.shape = N, C
  val -= val.max(axis=1).reshape(N, 1)
  loss = np.log(np.exp(val).sum(axis=1)).sum() - val[range(N), y].sum()

  counts = np.exp(val) / np.exp(val).sum(axis=1).reshape(N, 1)
  counts[range(N), y] -= 1
  dW = np.dot(X.T, counts)

  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


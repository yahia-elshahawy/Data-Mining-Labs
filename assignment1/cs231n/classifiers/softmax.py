import numpy as np
from random import shuffle
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
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W) # scores becomes of size 10 x 1, the scores for each class
    dscores = scores
    f_Yi = scores[y[i]] #score of true class
    f = scores
    # Dividing large numbers can be numerically unstable, so it is important to use a normalization trick
    # shift the values of f so that the highest number is 0
    logC = np.max(f) # logC that is used to avoid computation instability due to big numbers
    f -= logC # subtract max score from all scores
    f_Yi -= logC # subtract max score from true score too
    L_i = -np.log(np.exp(f_Yi) / np.sum(np.exp(f))) # safe to do, gives the correct answer
    
    # accumulate loss for the i-th example
    loss += L_i
    for j in range(num_classes):
        pj = np.exp(f[j])/np.sum(np.exp(f))
        dW[:,j] += (pj - (j == y[i]))*X[i]
    
       
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train # we want to average the gradient too
  
  # Add regularization to the loss.
  loss += .5 * reg * np.sum(W * W)
  dW += reg * W # regularize the weights (differentiate reguralized loss wrt W)
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
  num_examples = X.shape[0]
  # compute class scores for a linear classifier
  scores = np.dot(X, W)
  # to avoid big numbers computation instability
  scores -= scores.max(axis=1, keepdims=True)
  # get unnormalized probabilities
  exp_scores = np.exp(scores)
  # normalize them for each example
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  correct_logprobs = -np.log(probs[range(num_examples),y]) # we compute -log(exp(Yi)/sum(exp(yj))) those will only contribute to loss
  # compute the loss: average cross-entropy loss and regularization
  loss = np.sum(correct_logprobs)/num_examples # take averaged sum of all logs gained for each true class
  loss += 0.5*reg*np.sum(W*W)
  # for gradient calculations
  dscores = probs
  dscores[range(num_examples),y] -= 1 # subtract -1 from col where j = Yi (true score) 
  dscores /= num_examples # average
  dW = np.dot(X.T, dscores) # to get gradient we simply mul dscores to X
  dW += reg*W # don't forget the regularization gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


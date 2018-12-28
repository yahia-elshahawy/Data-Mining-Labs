import numpy as np
from random import shuffle

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
  delta = 1
  for i in range(num_train):
    wrong_classes_count = 0
    scores = X[i].dot(W) # scores becomes of size 10 x 1, the scores for each class
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        # skip for the true class to only loop over incorrect classes
        continue
      margin = scores[j] - correct_class_score + delta # note delta = 1
      if margin > 0: 
      # we accumilate loss only if margin is positive (margin > 0) that means that diff between score(j),true_score is < delta 
      # or score(j) > true_score            
        # accumulate loss for the i-th example
        loss += margin
        wrong_classes_count += 1
        dW[:, j] += X[i] # gradient for the rows where j≠yi (incorrect classes)
    # gradient for the rows where j=yi (correct classes)
    dW[:, y[i]] += - wrong_classes_count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train # we want to average the gradient too
  
  # Add regularization to the loss.
  loss += .5 * reg * np.sum(W * W)
  dW += reg * W # regularize the weights (differentiate reguralized loss wrt W)
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # scores becomes of size 10 x 1, the scores for each class
  
  # we want to have array of true scores given their indices >> y in each scores row(in each row means for each data sample) 
  correct_class_scores = scores[np.arange(num_train),y] 
  
  # to be able to subtract correct_class_scores[i] from all other class score for each row (data sample) 
  # we should transform correct_class_scores to matrix then transpose it to become column matrix 
  margins = np.maximum(0, scores - np.matrix(correct_class_scores).T + delta) 
  
  margins[np.arange(num_train),y] = 0 # set margin[i,yi] to 0 since we should ignore computation of margin when (yi == j)
  # to sum each row_values we use np.sum(mat, axis = 1) using axis = 1 to specify we want to sum col values on each row not summation of all matrix
  # np.mean will get us the average loss ( total loss for each sample / N )  
  loss = np.mean(np.sum(margins, axis=1)) 
  # to handle reguralization same as before  
  loss += 0.5 * reg * np.sum(W * W)        
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
    
  # we want to do the following
  # for each row (data sample) we set the col value if margin > 0 to 1 and count them(by summing since all other values are set to 0)
  # and the col value of the true class margin we set it to (- wrong classes count)
  
  gradient_coeffs = margins # take a copy of previously calculated margins
  gradient_coeffs[gradient_coeffs>0] = 1 # set all values that satisfy margin > 0 to 1 else set to 0
  wrong_classes_count = np.sum(gradient_coeffs,axis = 1) # vector that for each row(data sample xi) it's the count of classes having very close(less than delta) less scores than true class score, or higher score than true score (we used sum since all other classes that don't fulfill that condition is set to 0)
  gradient_coeffs[np.arange(num_train), y] = - wrong_classes_count.T
  dW = np.dot(X.T, gradient_coeffs) # 3073*500  . 500*10  =  3073 * 10
  
  dW /= num_train # len X is num of samples, we average each gradient
  dW += reg * W                         
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

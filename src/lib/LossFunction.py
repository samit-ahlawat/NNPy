from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
import numpy as np
import functools

class LossFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def loss(self, actual_output, predicted_output):
        raise NotImplementedError('Method needs to be implemented in class %s' % self.__class__.__name__)

    @abstractmethod
    def deriv(self, actual_output, predicted_output):
        raise NotImplementedError('Method needs to be implemented in class %s' % self.__class__.__name__)


class MeanSquaredError(LossFunction):
    def loss(self, actual_output, predicted_output):
        """
        Mean square loss.
        @param actual_output: N dimensional array of shape (#batches, , , ...).
        If multidimensional, will flatten the array to 2 dimensional array first
        @param predicted_output: N dimensional array of shape (#batches, , , ...).
        If multidimensional, will flatten the array to 2 dimensional array first
        @return: loss (float)
        """
        vec = np.subtract(actual_output, predicted_output)
        if len(vec.shape) > 2:
            vec = vec.reshape((vec.shape[0],-1))
        return np.einsum("ij,ij->", vec, vec) / (vec.shape[0] * vec.shape[1])

    def deriv(self, actual_input, predicted_input):
        return -np.subtract(actual_input, predicted_input)


class BinaryCrossEntropy(LossFunction):
    """ loss = -sum_i (y_i*log(y_i_hat) + (1-y_i)*log(1-y_i_hat))/# observations """
    def __init__(self, from_logits=False):
        self.fromLogits = from_logits

    def loss(self, actual_output, predicted_output):
        """
        Calculate loss
        @param actual_output: N dimensional actual y of shape (#batches, , , ...). Each element 0 or 1
        @param predicted_output: N dimensional predicted y of shape (#batches, , , ...). Each element between 0 and 1 if
        fromLogits = False (default).
        If fromLogits = True, the number is converted to a probability using 1 / (1 + exp(-y_hat))
        @return: Loss (float)
        """
        if self.fromLogits:
            predicted_output = 1.0 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
        res = np.multiply(actual_output, np.log(predicted_output)) + np.multiply(1.0 - actual_output, np.log(1.0 - predicted_output))
        size = functools.reduce(lambda x, y=1: x*y, actual_output.shape)
        return -res.sum()/size

    def deriv(self, actual_output, predicted_output):
        """
        Derivative = -(y_i - y_i_hat)/(y_i_hat * (1 - y_i_hat)) if fromLogits = False
        Derivative = -(y_i - y_i_hat) if fromLogits = True with y_i_hat = 1 / (1 + exp(-y_i_hat))
        @param actual_output:
        @param predicted_output:
        """
        if self.fromLogits:
            predicted_output = 1.0 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
            return np.subtract(predicted_output, actual_output)
        numerator = np.subtract(predicted_output, actual_output)
        denominator = np.multiply(predicted_output, 1.0 - predicted_output)
        return np.divide(numerator, denominator)


class CategoricalCrossEntropy(LossFunction):
    """ Categorical cross entropy from softmax. Probability of falling in one of many buckets.
    Expects a 0/1 encoding, with actual output 0 or 1
    fromLogits governs if the predicted output needs a softmax transformation or not
    """
    def __init__(self, from_logits=False):
        self.fromLogits = from_logits

    def loss(self, actual_output, predicted_output):
        """
        Calculate loss
        @param actual_output: 2 dimensional ndarray of shape (#batches, length of one output). Each entry must be 0 or 1
        @param predicted_output: 2 dimensional ndarray of shape (#batches, length of one output)
        @return Loss (float)
        """
        if self.fromLogits:
            exp_output = np.exp(predicted_output)
            sum_exp = exp_output.sum(axis=1)
            predicted_output = np.divide(exp_output, sum_exp[:, np.newaxis])
        size = actual_output.shape[0]
        return -np.einsum("ij,ij->", actual_output, np.log(predicted_output))/size

    def deriv(self, actual_output, predicted_output):
        if self.fromLogits:
            exp_output = np.exp(predicted_output)
            sum_exp = exp_output.sum(axis=1)
            predicted_output = np.divide(exp_output, sum_exp[:, np.newaxis])
            return -np.multiply(actual_output, 1.0 - predicted_output)
        return -np.divide(actual_output, predicted_output)


class SparseCategoricalCrossEntropy(CategoricalCrossEntropy):
    """ Similar to categorical cross entropy, except that actual outputs are indices of buckets and not 0/1 encoding """
    def __init__(self, from_logits=False):
        super(SparseCategoricalCrossEntropy, self).__init__(from_logits)

    def loss(self, actual_output, predicted_output):
        """
        Calculate loss
        @param actual_output: 1 dimensional ndarray of shape (#batches). Each entry must be index (0 based)
        of the correct category
        @param predicted_output: 2 dimensional ndarray of shape (#batches, length of one output). Length of output must be equal
        to the number of categories (buckets)
        @return Loss (float)
        """
        if self.fromLogits:
            exp_output = np.exp(predicted_output)
            sum_exp = exp_output.sum(axis=1)
            predicted_output = np.divide(exp_output, sum_exp[:, np.newaxis])
        size = predicted_output.shape[0]
        vals = -np.log(predicted_output)
        return vals[:, actual_output].sum()/size


    def deriv(self, actual_output, predicted_output):
        output = np.zeros(predicted_output.shape, dtype=np.bool)
        output[:, actual_output] = True
        return super(SparseCategoricalCrossEntropy, self).deriv(output, predicted_output)


class KLDivergence(LossFunction):
    """ Kullback Leibler Divergence """
    def __init__(self, from_logits=False, sparse=False):
        self.fromLogits = from_logits
        self.sparse = sparse

    def loss(self, actual_output, predicted_output):
        """
        Loss = sum_i(y_y * log(y_i/y_i_hat)) / #batches
        @param actual_output:
        @param predicted_output:
        """
        if self.fromLogits:
            exp_output = np.exp(predicted_output)
            sum_exp = exp_output.sum(axis=1)
            predicted_output = np.divide(exp_output, sum_exp[:, np.newaxis])
        if self.sparse:
            vals = predicted_output[:, actual_output]
            loss_vals = -np.log(vals)
            return loss_vals.sum()/actual_output.shape[0]
        # avoid log(0) for cases with 0 actual output
        log_act_output = np.where(actual_output == 0.0, 1.0, actual_output)
        log_term = np.log(np.divide(log_act_output, predicted_output))
        return np.einsum("ij,ij->", actual_output, log_term)/actual_output.shape[0]

    def deriv(self, actual_output, predicted_output):
        if self.fromLogits:
            exp_output = np.exp(predicted_output)
            sum_exp = exp_output.sum(axis=1)
            predicted_output = np.divide(exp_output, sum_exp[:, np.newaxis])
        if self.sparse:
            vals = np.zeros(predicted_output.shape, dtype=np.bool)
            vals[:, actual_output] = True
            actual_output = vals
        return -np.divide(actual_output, predicted_output)

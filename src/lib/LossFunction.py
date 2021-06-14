from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
import numpy as np


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
        assert actual_output.shape == predicted_output.shape
        vec = np.subtract(actual_output, predicted_output)
        if len(vec.shape) > 2:
            vec = vec.reshape((vec.shape[0], -1))
        return np.einsum("ij,ij->", vec, vec) / (vec.shape[0] * vec.shape[1])

    def deriv(self, actual_output, predicted_output):
        """
        Calculate derivative of loss w.r.t predicted output, i.e. DLoss/dy_i_t
        :param actual_output: N dimensional array of shape (#batches, , , ...).
        :param predicted_output: N dimensional array of shape (#batches, , , ...).
        :return: N dimensional array of shape (#batches, , , ...) containing derivative of loss w.r.t y for all batches
        """
        assert actual_output.shape == predicted_output.shape
        multiplier = -2.0 / actual_output.shape[0]
        return multiplier * np.subtract(actual_output, predicted_output)


class SparseBinaryCrossEntropy(LossFunction):
    """ loss = -sum_i (y_i*log(y_i_hat) + (1-y_i)*log(1-y_i_hat))/# observations """
    def __init__(self, from_logits=False):
        self.fromLogits = from_logits

    def loss(self, actual_output, predicted_output):
        """
        Calculate loss
        @param actual_output: 2 dimensional actual y of shape (#batches, 1). All elements are 0 or 1
        @param predicted_output: 2 dimensional predicted y of shape (#batches, 2). Each element between 0 and 1 if
        fromLogits = False (default).
        If fromLogits = True, the number is converted to a probability using 1 / (1 + exp(-y_hat))
        @return: Loss (float)
        """
        assert (actual_output == 0).sum() + (actual_output == 1).sum() == actual_output.shape[0]
        if self.fromLogits:
            max_val = np.max(predicted_output, axis=1)
            max_val = np.where(np.abs(max_val) < 1E-6, 1, max_val)
            predicted_output = np.divide(predicted_output, max_val[:, np.newaxis])
            predicted_output = 1.0 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
            predicted_output_sum = predicted_output.sum(axis=1)
            predicted_output = np.divide(predicted_output, predicted_output_sum[:, np.newaxis])
        res = np.ndarray(predicted_output.shape, dtype=np.float)
        res[:, 0] = -np.where(actual_output[:, 0] == 0, np.log(predicted_output[:, 0]), 0)
        res[:, 1] = -np.where(actual_output[:, 0] == 1, np.log(predicted_output[:, 1]), 0)
        return res.sum() / predicted_output.shape[0]

    def deriv(self, actual_output, predicted_output):
        """
        Derivative = -(y_i/y_i_hat + (1 - y_i)/(1 - y_i_hat)) if fromLogits = False
                   = -(y_i/y_i_hat) for both classes if fromLogits = False
        Derivative = -(y_i * (1 - y_i_hat) - (y_i_hat - y_i_hat**2)/(y_0_hat + y_1_hat))  if fromLogits = True with
                      y_i_hat = 1 / (1 + exp(-y_i_hat))
        @param actual_output: 2 dimensional nd array of shape (#batches, 1). Essentially a 1d array, but needs to be
        passed as 2d array
        @param predicted_output:
        """
        res = np.ndarray(predicted_output.shape, dtype=np.float)
        if self.fromLogits:
            max_val = np.max(predicted_output, axis=1)
            max_val = np.where(np.abs(max_val) < 1E-6, 1, max_val)
            predicted_output = np.divide(predicted_output, max_val[:, np.newaxis])
            predicted_output = 1.0 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
            p_sum = predicted_output.sum(axis=1)
            mat1 = np.subtract(1.0, predicted_output)
            mat2 = predicted_output - np.multiply(predicted_output, predicted_output)
            mat2 = -np.divide(mat2, p_sum[:, np.newaxis])
            res[:, :] = 0.0
            nrows = actual_output.shape[0]
            res[np.arange(nrows), actual_output[:, 0]] = mat1[np.arange(nrows), actual_output[:, 0]]
            res = np.add(res, mat2)
            return -res / nrows
        res[:, 0] = -np.where(actual_output[:, 0] == 0, np.divide(1.0, predicted_output[:, 0]), 0)
        res[:, 1] = -np.where(actual_output[:, 0] == 1, np.divide(1.0, predicted_output[:, 1]), 0)
        return res / actual_output.shape[0]


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
        assert actual_output.shape == predicted_output.shape
        if self.fromLogits:
            max_val = np.max(predicted_output, axis=1)
            min_val = np.min(predicted_output, axis=1)
            den = np.subtract(max_val, min_val)
            den = np.where(np.abs(den) < 1E-6, 1, den)
            num = np.subtract(predicted_output, min_val[:, np.newaxis])
            predicted_output = np.divide(num, den[:, np.newaxis])
            predicted_output = 1 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
            p_sum = predicted_output.sum(axis=1)
            predicted_output = np.divide(predicted_output, p_sum[:, np.newaxis])
        selected_cat = np.argmax(actual_output, axis=1)
        nrows = actual_output.shape[0]
        return -np.log(predicted_output[np.arange(nrows), selected_cat]).sum() / nrows

    def deriv(self, actual_output, predicted_output):
        assert actual_output.shape == predicted_output.shape
        res = np.zeros(predicted_output.shape, dtype=np.float)
        selected_cat = np.argmax(actual_output, axis=1)
        nrows = actual_output.shape[0]
        if self.fromLogits:
            max_val = np.max(predicted_output, axis=1)
            min_val = np.min(predicted_output, axis=1)
            den = np.subtract(max_val, min_val)
            den = np.where(np.abs(den) < 1E-6, 1, den)
            num = np.subtract(predicted_output, min_val[:, np.newaxis])
            predicted_output = np.divide(num, den[:, np.newaxis])
            predicted_output = 1 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
            p_sum = predicted_output.sum(axis=1)
            prob = np.divide(predicted_output, p_sum[:, np.newaxis])
            mat1 = np.subtract(1.0, predicted_output)
            mat2 = -np.multiply(prob, mat1)
            vect = mat1[np.arange(nrows), selected_cat]
            mat2[np.arange(nrows), selected_cat] += vect
            mat2 = np.divide(mat2, den[:, np.newaxis])
            return -mat2 / nrows
        res[np.arange(nrows), selected_cat] = np.divide(1.0, predicted_output[np.arange(nrows), selected_cat])
        return -res / nrows


class SparseCategoricalCrossEntropy(LossFunction):
    """ Similar to categorical cross entropy, except that actual outputs are indices of buckets and not 0/1 encoding """
    def __init__(self, from_logits=False):
        self.fromLogits = from_logits

    def loss(self, actual_output, predicted_output):
        """
        Calculate loss
        @param actual_output: 2 dimensional ndarray of shape (#batches, 1). Each entry must be index (0 based)
        of the correct category
        Can also be a 1 dimensional array of shape (#batches) containing the index of the correct category
        @param predicted_output: 2 dimensional ndarray of shape (#batches, length of one output). Length of output must be equal
        to the number of categories (buckets)
        @return Loss (float)
        """
        if len(actual_output.shape) == 2:
            actual_output = actual_output[:, 0]
        assert len(actual_output.shape) == 1
        assert len(predicted_output.shape) == 2
        if self.fromLogits:
            max_val = np.max(predicted_output, axis=1)
            min_val = np.min(predicted_output, axis=1)
            den = np.subtract(max_val, min_val)
            den = np.where(np.abs(den) < 1E-6, 1, den)
            num = np.subtract(predicted_output, min_val[:, np.newaxis])
            predicted_output = np.divide(num, den[:, np.newaxis])
            predicted_output = 1 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
            p_sum = predicted_output.sum(axis=1)
            predicted_output = np.divide(predicted_output, p_sum[:, np.newaxis])
        nrows = actual_output.shape[0]
        return -np.log(predicted_output[np.arange(nrows), actual_output]).sum() / nrows

    def deriv(self, actual_output, predicted_output):
        if len(actual_output.shape) == 2:
            actual_output = actual_output[:, 0]
        assert len(actual_output.shape) == 1
        nrows = actual_output.shape[0]
        if self.fromLogits:
            max_val = np.max(predicted_output, axis=1)
            min_val = np.min(predicted_output, axis=1)
            den = np.subtract(max_val, min_val)
            den = np.where(np.abs(den) < 1E-6, 1, den)
            num = np.subtract(predicted_output, min_val[:, np.newaxis])
            predicted_output = np.divide(num, den[:, np.newaxis])
            predicted_output = 1 + np.exp(-predicted_output)
            predicted_output = np.divide(1.0, predicted_output)
            p_sum = predicted_output.sum(axis=1)
            prob = np.divide(predicted_output, p_sum[:, np.newaxis])
            mat1 = np.subtract(1.0, predicted_output)
            mat2 = -np.multiply(prob, mat1)
            vect = mat1[np.arange(nrows), actual_output]
            mat2[np.arange(nrows), actual_output] += vect
            mat2 = np.divide(mat2, den[:, np.newaxis])
            return -mat2 / nrows
        res = np.zeros(predicted_output.shape, dtype=np.float)
        res[:, actual_output] = np.divide(1.0, predicted_output[:, actual_output])
        return -res


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
        return -np.divide(actual_output, predicted_output).sum(axis=0) / actual_output.shape[0]

from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
import numpy as np


class Metric(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def value(self, actual_output, predicted_output, **kwargs):
        raise NotImplementedError('Method needs to be implemented in class %s' % self.__class__.__name__)


class Accuracy(Metric):
    def __init__(self, probability=False, sparse=True):
        """
        Initialize
        :param probability: True or False. If True, predicted_output is a probability of falling in a category. To
        compare it against output, bucket with maximum probability is considered.
        """
        self.isProbability = probability
        self.isSparse = sparse

    def value(self, actual_output, predicted_output, **kwargs):
        """
        Fraction of inputs correctly classified
        :param actual_output: N dimensional array of shape (#batches, ...)
        if isProbability = True, 2 dimensional array of shape (#batches, 1) if isSparse = True
        if isProbability = True, 2 dimensional array of shape (#batches, #buckets) if isSparse = False
        :param predicted_output: N dimensional array of shape (#batches, ...)
        if isProbability = True, 2 dimensional array of shape (#batches, #buckets)
        :param kwargs:
        :return:
        """
        if self.isProbability:
            predicted_output = np.argmax(predicted_output, axis=1)
            predicted_output = predicted_output[:, np.newaxis]
            if not self.isSparse:
                actual_output = np.argmax(actual_output, axis=1)
        return np.equal(actual_output, predicted_output).sum() / float(actual_output.shape[0])


class Precision(Metric):
    """ true positive / (true positive + false positive) """
    def __init__(self, positive_prediction, positive_observation=None):
        """
        Initialize
        :param positive_prediction: method that takes prediction and return True or False depending on
        whether the prediction is positive or negative
        :param positive_observation: (optional) method that takes observation and return True or False depending on
        whether the observation is positive or negative. If none, uses the same method as positive_prediction
        """
        self.positive = positive_prediction
        self.positiveObs = positive_observation
        if positive_observation is None:
            self.positiveObs = self.positive

    def value(self, actual_output, predicted_output, **kwargs):
        positives = 0  # includes true positive and false positives
        true_positives = 0
        for i in range(actual_output.shape[0]):
            pos_obs = self.positiveObs(actual_output[i, :])
            pos_pred = self.positive(predicted_output[i, :])
            if pos_pred:
                if pos_obs:
                    true_positives += 1
                positives += 1
        if positives == 0:
            return 0
        return float(true_positives)/positives


class Recall(Metric):
    """ true positive / (true positive + false negative)"""
    def __init__(self, positive_prediction, positive_observation=None):
        """
        Initialize
        :param positive_prediction: method that takes prediction and return True or False depending on
        whether the prediction is positive or negative
        :param positive_observation: (optional) method that takes observation and return True or False depending on
        whether the observation is positive or negative. If none, uses the same method as positive_prediction
        """
        self.positive = positive_prediction
        self.positiveObs = positive_observation
        if positive_observation is None:
            self.positiveObs = self.positive

    def value(self, actual_output, predicted_output, **kwargs):
        positives = 0  # includes true positive and false negatives
        true_positives = 0
        for i in range(actual_output.shape[0]):
            pos_obs = self.positiveObs(actual_output[i, :])
            pos_pred = self.positive(predicted_output[i, :])
            if pos_obs:
                if pos_pred:
                    true_positives += 1
                positives += 1
        if positives == 0:
            return 0
        return float(true_positives)/positives


class AUC(Metric):
    """
    Area under ROC plot of true positive rate against false positive rate
    True positive rate = true positive / (true positive + false negative)
    False positive rate = false positive / (false positive + true negative)
    """
    def __init__(self, positive_prediction, positive_observation=None):
        """
        Initialize
        :param positive_prediction: method that takes prediction and return True or False depending on
        whether the prediction is positive or negative
        :param positive_observation: (optional) method that takes observation and return True or False depending on
        whether the observation is positive or negative. If none, uses the same method as positive_prediction
        """
        self.positive = positive_prediction
        self.positiveObs = positive_observation
        if positive_observation is None:
            self.positiveObs = self.positive

    def value(self, actual_output, predicted_output, **kwargs):
        area, last_tpr, last_fpr = 0.0, 0.0, 0.0
        true_positives, positive_preds, all_positives, all_negatives = 0, 0, 0, 0
        result = []
        for i in range(len(actual_output.shape[0])):
            obs = self.positiveObs(actual_output[i, :])
            if obs:
                all_positives += 1
            else:
                all_negatives += 1

            if self.positive(predicted_output[i, :]):
                if obs:
                    true_positives += 1
                positive_preds += 1
                result.append((true_positives, positive_preds))

        for tp, pos_pred in result:
            tpr = float(tp)/all_positives
            fpr = float(pos_pred - tp)/all_negatives
            area += (fpr - last_fpr) * last_tpr
            last_tpr, last_fpr = tpr, fpr

        area += (1.0 - last_fpr) * last_tpr
        return area
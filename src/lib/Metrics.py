from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
import numpy as np
import functools

class Metric(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def value(self, actual_output, predicted_output):
        raise NotImplementedError('Method needs to be implemented in class %s' % self.__class__.__name__)


class Accuracy(Metric):
    """ Fraction of inputs correctly classified """
    def value(self, actual_output, predicted_output):
        size = functools.reduce(lambda x,y=1: x*y, predicted_output.shape)
        return (actual_output == predicted_output).sum()/float(size)

class Precision(Metric):
    def value(self, actual_output, predicted_output):
        pass

class Recall(Metric):
    def value(self, actual_output, predicted_output):
        pass

class AUC(Metric):
    def value(self, actual_output, predicted_output):
        pass

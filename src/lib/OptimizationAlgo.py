from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
import numpy as np


class OptimizationAlgo(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getCorrections(self, gradient):
        raise NotImplementedError('Method needs to be implemented in class %s' % self.__class__.__name__)


class SimpleGradDescent(OptimizationAlgo):
    """ Simple gradient descent optimizer """
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self.alpha = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = 0
        self.velocityList = None

    def getCorrections(self, gradient):
        self.velocity = self.momentum * self.velocity - self.alpha * gradient
        if self.nesterov:
            return self.momentum * self.velocity - self.alpha * gradient
        return self.velocity


class RMSProp(OptimizationAlgo):
    """ RMSProp: keeps a moving average of square of gradients as a variance estimate """
    def __init__(self, learning_rate=0.01, rho=0.9, epsilon=1E-6):
        self.alpha = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.variance = 0

    def getCorrections(self, gradient):
        self.variance = self.rho * self.variance + (1 - self.rho) * np.multiply(gradient, gradient)
        val = np.sqrt(self.variance + self.epsilon)
        return -self.alpha * np.divide(gradient, val)


class ADAM(OptimizationAlgo):
    """ Adaptive Moment Estimation. Uses a moving average for gradient and gradient^2 """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1E-6):
        self.alpha = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment = 0
        self.variance = 0
        self.beta1t = beta1
        self.beta2t = beta2

    def getCorrections(self, gradient):
        self.moment = self.beta1 * self.moment + (1 - self.beta1) * gradient
        self.variance = self.beta2 * self.variance + (1 - self.beta2) * np.multiply(gradient, gradient)
        moment_hat = np.divide(self.moment, 1 - self.beta1t)
        variance_hat = np.divide(self.variance, 1 - self.beta2t)
        den = np.sqrt(variance_hat) + self.epsilon
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        return -self.alpha * np.divide(moment_hat, den)




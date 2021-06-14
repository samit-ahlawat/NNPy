from __future__ import absolute_import, division, print_function
import numpy as np
from abc import ABCMeta, abstractmethod


class Activation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def output(self, inputs):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def deriv(self, inputs):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)


class UnitActivation(Activation):
    """ output = x (same as input) """

    def output(self, inputs):
        return inputs

    def deriv(self, inputs):
        return np.zeros(inputs.shape)


class Sigmoid(Activation):
    """ 1/(1 + exp(-x)) """
    def output(self, inputs):
        """
        Find output 1/(1 + exp(-inputs))
        :param inputs:
        :return: 1/(1 + exp(-x*beta -bias))
        """
        val = np.exp(-inputs)
        return np.divide(1.0, np.add(1.0, val))

    def deriv(self, inputs):
        """
        Derivative of the activation function with respect to weights
        :param inputs: This is weights * inputs + bias from a neuron that is sent to activation function
        :return: derivative
        """
        act_val = self.output(inputs)
        return np.multiply(act_val, np.subtract(1.0, act_val))


class HyperbolicTangent(Activation):
    """ (exp(x) - exp(-x))/(exp(x) + exp(-x)) """
    def output(self, inputs):
        val = np.exp(2*inputs)
        return np.divide(val - 1, val + 1)

    def deriv(self, inputs):
        """
        Derivative of the activation function with respect to weights
        :param inputs: This is weights * inputs + bias from a neuron that is sent to activation function
        :return: derivative
        """
        val = np.divide(1.0, np.exp(2 * inputs) + 1)
        return np.multiply(4*val, 1-val)


class RectifiedLinear(Activation):
    """ max(0, x) """
    def output(self, inputs):
        return max(0, inputs)

    def deriv(self, inputs):
        """
        Derivative of the activation function with respect to weights
        :param inputs: This is weights * inputs + bias from a neuron that is sent to activation function
        :return: derivative
        """
        return np.where(inputs <= 0, 0.0, 1.0)


class Exponential(Activation):
    """ exp(x) """
    def output(self, inputs):
        return np.exp(inputs)

    def deriv(self, inputs):
        """
        Derivative of the activation function with respect to weights
        :param inputs: This is weights * inputs + bias from a neuron that is sent to activation function
        :return: derivative
        """
        return np.exp(inputs)


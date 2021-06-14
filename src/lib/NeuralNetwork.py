from __future__ import absolute_import, division, print_function
import numpy as np
from abc import ABCMeta, abstractmethod
import functools
import logging
import src.lib.OptimizationAlgo as OA
import src.lib.LossFunction as LF
import src.lib.Metrics as MT
from src.lib.Layer import Layer, DropoutLayer
from copy import copy


FORMAT = '%(asctime)s %(relativeCreated)6d %(threadName)s %(name)s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class Network(object):
    __metaclass__ = ABCMeta

    def __init__(self, loss_func=LF.MeanSquaredError(),  optim_algo=OA.SimpleGradDescent(), metrics=[],
                 log_level=logging.DEBUG):
        assert isinstance(loss_func, LF.LossFunction)
        self.lossFunc = loss_func
        assert isinstance(optim_algo, OA.OptimizationAlgo)
        self.optimAlgo = optim_algo
        for metric in metrics:
            assert isinstance(metric, MT.Metric)
        self.metrics = metrics
        self.dropoutLayers = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    @abstractmethod
    def setWeights(self, params):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def getWeights(self):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def predict(self, inputs, bias_for_layers=None, **kwargs):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def fit(self, inputs, outputs, bias_for_layers=None, epochs=10, **kwargs):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    def addMetrics(self, metrics):
        self.metrics.extend(metrics)

    @abstractmethod
    def summary(self):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)


class SequentialNeuralNetwork(Network):
    """ Sequential neural network composed of sequential layers """

    def __init__(self, loss_func=LF.MeanSquaredError(),  optim_algo=OA.SimpleGradDescent(), metrics=[], layers=[],
                 log_level=logging.DEBUG):
        """
        Initialize the sequential network
        @param loss_func: Loss function. Must be of type LossFunction
        @param optim_algo:Optimization algorithm. Must be of type OptimizationAlgo
        @param metrics: a list of metrics, each of type src.lib.Metric.Metric
        @param layers: a list of layers forming the sequential network. Begins with the input layer.
        Each layer must be of type src.lib.Layer.Layer
        @param log_level: Level of logging (logging.DEBUG default)
        """
        super(SequentialNeuralNetwork, self).__init__(loss_func, optim_algo, metrics, log_level)
        for layer in layers:
            assert isinstance(layer, Layer)
        self.layers = layers
        self.dropoutLayers = [lyr for lyr in self.layers if isinstance(lyr, DropoutLayer)]
        self.oAlgos = None


    def addLayer(self, layer):
        """
        Add a layer to the network. (adds on top of the existing layers)
        @param layer: a layer object of type src.lib.Layer.Layer
        """
        assert isinstance(layer, Layer)
        self.layers.append(layer)
        if isinstance(layer, DropoutLayer):
            self.dropoutLayers.append(layer)

    def setWeights(self, weights):
        """
        Set weights for layers
        @param weights: list or tuple containing weights for each layer
        """
        for i, weight in enumerate(weights):
            self.layers[i].setWeights(weight)

    def getWeights(self):
        """
        Get weights for all layers
        @return: list of weights for each layer
        """
        return [layer.getWeights() for layer in self.layers]

    def predict(self, inputs, bias_for_layers=None, **kwargs):
        """
        Predict the output for provided inputs and layer biases
        @param inputs: n dimensional array of shape (#batches, , , )
        @param bias_for_layers: list of arrays, one for each layer, or None
        @param kwargs: get_all_layer_outputs=True to get output of all layers in the network (list of outputs, one for each layer)
        @return: output of last layer, or output of all layers if get_all_layer_outputs=True
        """
        if not self.layers:
            return inputs
        if not kwargs.get("training", False):
            for dropout_lyr in self.dropoutLayers:
                dropout_lyr.TRAINING = False
        if bias_for_layers is None:
            bias_for_layers = [None] * len(self.layers)

        outputs = [None] * len(self.layers)
        outputs[0] = self.layers[0].output(inputs, bias=bias_for_layers[0])
        for i, layer in enumerate(self.layers[1:]):
            outputs[i+1] = layer.output(outputs[i], bias=bias_for_layers[i+1])

        if kwargs.get("get_all_layer_outputs", False):
            return outputs
        return outputs[-1]


    def getWeightCorrections(self, weight_gradients):
        """
        Get weight corrections for weight gradients, using the optimization algorithm specified in constructor
        @param weight_gradients: Weight gradients. Can be ndarray or a tuple (or list) of ndarrays
        @return: weight corrections (ndarray or a list of ndarrays, according to the type of weight_gradients)
        """
        if isinstance(weight_gradient, (tuple, list)):
            weight_corrections = [None] * len(weight_gradients)
            if self.oAlgos is None:
                self.oAlgos = [copy(self.optimAlgo) for i in range(len(weight_gradients))]
            for i, weight_grad in enumerate(weight_gradients):
                weight_corrections[i] = self.oAlgos[i].getCorrections(weight_grad)
            return weight_corrections

        return self.optimAlgo.getCorrections(weight_gradient)


    def fit(self, inputs, outputs, bias_for_layers=None, epochs=10, **kwargs):
        """
        Train the network to fit the data
        @param inputs: input ndarray of shape (#batches, ...)
        @param outputs: output ndarray of shape (#batches, ...)
        @param bias_for_layers: list of biases to apply for each layer. None for no bias (default)
        @param epochs: Number of epochs for training
        @param kwargs:
        @return: dictionary containing the loss and metrics for each epoch
        """
        for droupout_lyr in self.dropoutLayers:
            droupout_lyr.TRAINING = True

        result = {"epoch":[], "loss":[]}
        for metric in self.metrics:
            result[metric.__class__.__name__] = []

        for epoch in range(epochs):
            predicted_outputs = self.predict(inputs, bias_for_layers, get_all_layer_outputs=True, training=True)
            output_shape = predicted_outputs[-1].shape[1:]
            if len(output_shape) == 1:
                output_shape = output_shape[0]
            delta_arr_layer = np.ones(output_shape, dtype=np.float)
            deriv_loss_wrt_outputs = self.lossFunc.deriv(outputs, predicted_outputs[-1])
            for i in range(len(self.layers)-1, 0, -1):
                layer = self.layers[i]
                layer_inputs = predicted_outputs[i-1]
                layer_deriv = layer.deriv(layer_inputs, bias_for_layers[i])
                weight_gradient, delta_arr_lastlayer = layer.backPropagation(layer_inputs,
                                                                             layer_deriv,
                                                                             delta_arr_layer,
                                                                             deriv_loss_wrt_outputs,
                                                                             bias=bias_for_layers[i])
                weight_corrections = self.getWeightCorrections(weight_gradient)
                layer.applyWeightCorrections(weight_corrections)
                delta_arr_layer = delta_arr_lastlayer

            # record loss and metrics
            result["epoch"].append(epoch)
            loss = self.lossFunc.loss(outputs, predicted_outputs[-1])
            result["loss"].append(loss)
            for metric in self.metrics:
                value = metric.value(outputs, predicted_outputs[-1])
                result[metric.__class__.__name__].append(value)
            self.logger.info("Epoch: %d, loss: %f" % (epoch, loss))

        return result

    def summary(self):
        self.logger.info("%d layers", len(self.layers))



from __future__ import absolute_import, division, print_function
import numpy as np
from abc import ABCMeta, abstractmethod
import logging
import src.lib.OptimizationAlgo as OA
import src.lib.LossFunction as LF
import src.lib.Metrics as MT
from src.lib.History import History
from src.lib.Layer import Layer, DropoutLayer
from copy import copy
import functools

FORMAT = '%(asctime)s %(relativeCreated)6d %(threadName)s %(name)s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class Network(object):
    __metaclass__ = ABCMeta

    def __init__(self, loss_func=LF.MeanSquaredError(),  optim_algo=OA.SimpleGradDescent(), metrics=(),
                 log_level=logging.DEBUG):
        assert isinstance(loss_func, LF.LossFunction)
        self.lossFunc = loss_func
        assert isinstance(optim_algo, OA.OptimizationAlgo)
        self.optimAlgo = optim_algo
        for metric in metrics:
            assert isinstance(metric, MT.Metric)
        self.metrics = list(metrics)
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

    def __init__(self, loss_func=LF.MeanSquaredError(),  optim_algo=OA.SimpleGradDescent(), metrics=(), layers=(),
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
        self.layers = list(layers)
        self.dropoutLayers = [lyr for lyr in layers if isinstance(lyr, DropoutLayer)]
        self.oAlgos = [copy(optim_algo) for i in range(len(layers))]

    def addLayer(self, layer):
        """
        Add a layer to the network. (adds on top of the existing layers)
        @param layer: a layer object of type src.lib.Layer.Layer
        """
        assert isinstance(layer, Layer)
        self.layers.append(layer)
        if isinstance(layer, DropoutLayer):
            self.dropoutLayers.append(layer)
        self.oAlgos.append(copy(self.optimAlgo))

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
        training = kwargs.get("training", False)
        for dropout_lyr in self.dropoutLayers:
            dropout_lyr.TRAINING = training
        if bias_for_layers is None:
            bias_for_layers = [None] * len(self.layers)

        outputs = [None] * len(self.layers)
        outputs[0] = self.layers[0].output(inputs, bias=bias_for_layers[0])
        for i, layer in enumerate(self.layers[1:]):
            outputs[i+1] = layer.output(outputs[i], bias=bias_for_layers[i+1])

        if kwargs.get("get_all_layer_outputs", False):
            return outputs
        return outputs[-1]

    def getWeightCorrections(self, weight_gradients, layer_number):
        """
        Get weight corrections for weight gradients, using the optimization algorithm specified in constructor
        @param weight_gradients: Weight gradients. Can be ndarray or a tuple (or list) of ndarrays
        @param layer_number: Index of this layer. Must be an integer between 0 and nLayers-1
        @return: weight corrections (ndarray or a list of ndarrays, according to the type of weight_gradients)
        """
        if isinstance(weight_gradients, (tuple, list)):
            weight_corrections = [None] * len(weight_gradients)
            if not isinstance(self.oAlgos[layer_number], list):
                self.oAlgos[layer_number] = [copy(self.optimAlgo) for i in range(len(weight_gradients))]
            for i, weight_grad in enumerate(weight_gradients):
                weight_corrections[i] = self.oAlgos[layer_number][i].getCorrections(weight_grad)
            return weight_corrections

        return self.oAlgos[layer_number].getCorrections(weight_gradients)

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
        if len(outputs.shape) == 1:
            outputs = outputs[:, np.newaxis]

        loss_cols = [self.lossFunc.__class__.__name__] + [metric.__class__.__name__ for metric in self.metrics]
        result = History(loss_cols)

        if bias_for_layers is None:
            bias_for_layers = [None] * len(self.layers)

        init_arr, init_delta_arr = None, None
        for epoch in range(epochs):
            predicted_outputs = self.predict(inputs, bias_for_layers, get_all_layer_outputs=True, training=True)
            output_shape = predicted_outputs[-1].shape[1:]
            if init_arr is None:
                flat_shape = functools.reduce(lambda x, y=1: x*y, output_shape)
                init_arr = np.eye(flat_shape, dtype=np.float).reshape(output_shape + output_shape)
                init_delta_arr = np.zeros((outputs.shape[0],) + output_shape + output_shape, dtype=np.float)
                init_delta_arr[:, ...] = init_arr
            delta_arr_layer = init_delta_arr
            deriv_loss_wrt_outputs = self.lossFunc.deriv(outputs, predicted_outputs[-1])
            for i in range(len(self.layers)-1, -1, -1):
                layer = self.layers[i]
                if i == 0:
                    layer_inputs = inputs
                else:
                    layer_inputs = predicted_outputs[i-1]
                layer_deriv = layer.deriv(layer_inputs, bias_for_layers[i])
                weight_gradient, delta_arr_lastlayer = layer.backPropagation(layer_inputs,
                                                                             layer_deriv,
                                                                             delta_arr_layer,
                                                                             deriv_loss_wrt_outputs,
                                                                             bias=bias_for_layers[i])
                weight_corrections = self.getWeightCorrections(weight_gradient, i)
                layer.applyWeightCorrections(weight_corrections)
                delta_arr_layer = delta_arr_lastlayer

            # record loss and metrics
            loss = self.lossFunc.loss(outputs, predicted_outputs[-1])
            metric_vals = [loss]
            for metric in self.metrics:
                value = metric.value(outputs, predicted_outputs[-1])
                metric_vals.append(value)
            result.append(epoch+1, metric_vals)
            self.logger.info("Epoch: %d, loss: %f, metrics: %s" % (epoch+1, loss, str(metric_vals)))

        return result

    def summary(self):
        self.logger.info("%d layers", len(self.layers))



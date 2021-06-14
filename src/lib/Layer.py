from __future__ import absolute_import, division, print_function
import numpy as np
from abc import ABCMeta, abstractmethod
from enum import IntEnum, unique
from src.lib.Activation import UnitActivation, Activation, Sigmoid, HyperbolicTangent
import functools


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def output(self, inputs, bias=None, **kwargs):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def deriv(self, inputs, bias=None):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def applyWeightsToInput(self, inputs, bias=None):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def inputShape(self):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def outputShape(self, input_shape=None):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def applyWeightCorrections(self, weight_correction):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def getWeights(self):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)

    @abstractmethod
    def setWeights(self, weights):
        raise NotImplementedError('Needs implementation in %s' % self.__class__.__name__)


class DenseLayer(Layer):
    """ Dense layer with each neuron connected to each neuron or input from the previous layer """

    def __init__(self, num_inputs, num_neurons, network_output_neurons, activation=UnitActivation(), init_wt=1E-2):
        """
        Initialize the dense layer. Last weight is for bias
        @param num_inputs: Number of inputs to this layer
        @param num_neurons: Number of neurons in this layer
        @param network_output_neurons: Number of outputs from the neural network (number of neurons in output layer of network)
        @param activation: activation function
        @param init_wt: initial weight multiplier, initial weights between 0 and init_wt
        """
        assert isinstance(activation, Activation)
        self.nInput = num_inputs
        self.nOutput = num_neurons
        self.activation = activation
        # including bias weight. Last weight is for bias
        self.weights = np.random.random((self.nOutput, self.nInput + 1)) * init_wt
        self.networkOutputNrs = network_output_neurons

    def applyWeightsToInput(self, inputs, bias=None):
        """
        Calculate weight*inputs + bias
        @param inputs: 2 dimensional array: (#batches, input dimension)
        @param bias: optional. 0 if not specified. Else, 2 dimensinal array of shape (#batches, #neurons)
        @return: weight*inputs + bias
        """
        x = np.einsum("ij,kj->ik", inputs, self.weights[:, 0:-1])
        if bias:
            x += np.einsum("ij,j->ij", bias, self.weights[:, -1])
        return x

    def output(self, inputs, bias=None, **kwargs):
        """
        Output of this layer
        @param inputs: numpy ndarray (2 dimensions): #batches X #inputs
        @param bias: numpy ndarray (1 dimension): #neurons i.e. number of outputs. Bias applied to each neuron. (optional)
        @return: output of layer: numpy ndarray (2 dimensions): # batches X #neurons
        """
        x = self.applyWeightsToInput(inputs, bias)
        return self.activation.output(x)

    def deriv(self, inputs, bias=None):
        """
        Derivative of activation function for each neuron
        @param inputs: numpy ndarray (2 dimensions): #batches X input dimension
        @param bias: numpy ndarray (1 dimension): #neurons. Bias applied to each neuron. (optional)
        @return: numpy ndarray (2 dimensions): #batches X #neurons. Derivative of activation function for each neuron
        """
        x = self.applyWeightsToInput(inputs, bias)
        return self.activation.deriv(x)

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation algorithm for the layer
        @param inputs: numpy ndarray (2 dimensions): #batches X #inputs
        @param layer_deriv: derivative of activation function for each neuron: d(activation_func). 2 dimensional: #batches X #neurons in layer
        @param delta_arr_layer: 3 dimensional ndarray containing dy_{network_output_neuron}/dy_j.
        dimensions: (#batches, #network output neurons, #neurons in this layer)
        This must be provided
        @param deriv_loss_wrt_outputs: dLoss/dy_{network_output_neurons} : 2 dimensional ndarray of shape (# batches, # output_neurons_in_network)
        @param bias: optional. 0 if not specified. Else, 2 dimensional array of shape (#batches, #neurons)
        @return: ndarray with dLoss/dweight_{i,j}: 2 dimensional ndarray: #outputs X #inputs, i.e. gradient for each weight and
        delta_arr_lastlayer: This is calculated by backpropagation. It contains dy_{output_neuron}/dy_k for y_k outputs from
        previous layer that are inputs to this layer. Dimensions: (#batches, #output_neurons, input dimension from last layer)
        """
        delta_arr_lastlayer = np.einsum("ijk,ik,kl->ijl", delta_arr_layer, layer_deriv, self.weights[:, 0:-1])
        # calculate weight gradients
        weight_gradients = np.ndarray(self.weights.shape, dtype=np.float)
        weight_gradients[:, 0:-1] = np.einsum("ij,ijk,ik,il->kl", deriv_loss_wrt_outputs,
                                              delta_arr_layer,
                                              layer_deriv,
                                              inputs)
        weight_gradients[:, -1] = 0.0
        if bias:
            weight_gradients[:, -1] = np.einsum("ij,ijk,ik,ik->k", deriv_loss_wrt_outputs,
                                                delta_arr_layer,
                                                layer_deriv,
                                                bias)
        return weight_gradients, delta_arr_lastlayer

    def outputShape(self, input_shape=None):
        """
        Shape of output: (#batches, Number of neurons in the layer)
        @return: (None, #neurons)
        """
        return None, self.nOutput

    def inputShape(self):
        """
        Shape of input: (#batches, Number of inputs to this layer)
        @return: None, #inputs
        """
        return None, self.nInput

    def applyWeightCorrections(self, weight_correction):
        self.weights = np.add(self.weights, weight_correction)

    def getWeights(self):
        """
        Get weights. Returns a reference, changing the returned value will directly change the layer weights.
        @return: Reference to layer weights
        """
        return self.weights

    def setWeights(self, weights):
        """
        Set layer weights
        @param weights: 2 dimensional ndarray of same shape as weights
        """
        self.weights[:, :] = weights


class SparseLayer(DenseLayer):
    """ Sparse layer with each neuron connected to specified neurons or inputs from the previous layer """
    def __init__(self, num_inputs, connections, network_output_neurons, activation=UnitActivation()):
        """
        Initialize the sparse layer
        @param num_inputs: Number of inputs to this layer
        @param connections: List of lists. Each list specified the connections for a neuron in the layer (from prev layer or inputs)
        E.g. [[0,1], [2]] represents a sparse layer with 2 neurons. First neuron is connected to first and second neurons from previous layer
        while last neuron is connected to last (3rd) neuron from previous layer.
        @param network_output_neurons: Number of outputs from the neural network (number of neurons in output layer of network)
        @param activation: activation function
        """
        super(SparseLayer, self).__init__(num_inputs, len(connections), network_output_neurons, activation)
        self.connections = connections
        self.indicator = np.zeros((self.nOutput, self.nInput), dtype=bool)
        for i in range(self.nOutput):
            self.indicator[i, self.connections[i]] = True
        self.weights[:, 0:-1] = np.multiply(self.weights[:, 0:-1], self.indicator)

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation algorithm for the layer
        @param inputs: numpy ndarray (2 dimensions): #batches X #inputs
        @param layer_deriv: derivative of activation function for each neuron: d(activation_func). 2 dimensional: #batches X #neurons in layer
        @param delta_arr_layer: 3 dimensional ndarray containing dy_{network_output_neuron}/dy_j.
        dimensions: (#batches, #network output neurons, #neurons in this layer)
        This must be provided
        @param deriv_loss_wrt_outputs: dLoss/dy_{network_output_neurons} : 2 dimensional ndarray of shape (#batches,  #output_neurons_in_network)
        @param bias: optional. 0 if not specified. Else, 2 dimensinal array of shape (#batches, #neurons)
        @return: ndarray with dLoss/dweight_{i,j}: 2 dimensional ndarray: #outputs X #inputs, i.e. gradient for each weight and
        delta_arr_lastlayer: This is calculated by backpropagation. It contains dy_{output_neuron}/dy_k for y_k outputs from
        previous layer that are inputs to this layer. Dimensions: (#batches, #output_neurons, input dimension from last layer)
        """
        delta_arr_lastlayer = np.einsum("ijk,ik,kl->ijl", delta_arr_layer, layer_deriv, self.weights[:, 0:-1])
        # calculate weight gradients
        weight_gradients = np.ndarray(self.weights.shape, dtype=np.float)
        weight_gradients[:,0:-1] = np.einsum("ij,ijk,ik,il,kl->kl", deriv_loss_wrt_outputs,
                                             delta_arr_layer,
                                             layer_deriv,
                                             inputs,
                                             self.indicator)
        weight_gradients[:, -1] = 0.0
        if bias:
            weight_gradients[:, -1] = np.einsum("ij,ijk,ik,ik->k", deriv_loss_wrt_outputs,
                                                delta_arr_layer,
                                                layer_deriv,
                                                bias)
        return weight_gradients, delta_arr_lastlayer

    def setWeights(self, weights):
        """
        Set layer weights
        @param weights: 2 dimensional ndarray of same shape as weights
        """
        self.weights[:, 0:-1] = np.multiply(weights[:, 0:-1], self.indicator)
        self.weights[:, -1] = weights[:, -1]


@unique
class PoolingType(IntEnum):
    MAX = 0,
    AVG = 1


class PoolingLayer3D(Layer):
    """
    3D Pooling layer for pooling the output of a convolution layer. Layer has no trainable weights.
    Can be used as a 2D pooling layer by providing pooling depth as 1
    """

    POOL_FUNC = [np.max, np.mean]

    POOL_DERIV_FUNC = ['maxDeriv', 'meanDeriv']

    def __init__(self, pooling_shape, pooling_type=PoolingType.MAX):
        """
        Initialize the pooling layer
        :@param pooling_shape: (pooled_len, pooled_width, pooling_depth) tuple
        :@param pooling_type: PoolingType
        """
        assert isinstance(pooling_type, PoolingType)
        assert len(pooling_shape) == 3
        self.poolingShape = pooling_shape
        self.poolingType = pooling_type

    def maxDeriv(self, inputs):
        """
        Derivative of max pooling layer
        :@param inputs: ndarray of (#batches, len, width, #channels) shape
        :@return: derivative of shape (#batches, len, width, #channels)
        """
        output_shape = (inputs.shape[0],
                        inputs.shape[1] // self.poolingShape[0],
                        inputs.shape[2] // self.poolingShape[1],
                        inputs.shape[3] // self.poolingShape[2])
        pool_shape = self.poolingShape
        rows_list = [range(i * pool_shape[0], (i + 1) * pool_shape[0]) for i in range(output_shape[1])]
        cols_list = [range(i * pool_shape[1], (i + 1) * pool_shape[1]) for i in range(output_shape[2])]
        dep_list = [range(i * pool_shape[2], (i + 1) * pool_shape[2]) for i in range(output_shape[3])]
        max_val = np.max(inputs[:, rows_list, :, :][:, :, :, cols_list, :][:, :, :, :, :, dep_list], axis=(2, 4, 6))
        val = (inputs[:, rows_list, :, :][:, :, :, cols_list, :][:, :, :, :, :, dep_list] == max_val[:, :, np.newaxis, :, np.newaxis, :, np.newaxis])
        val = np.where(val, 1, 0)
        return val.reshape(inputs.shape)

    def meanDeriv(self, inputs):
        """
        Derivative of mean pooling layer
        :@param inputs: ndarray of (#batches, len, width, #channels) shape
        :@return: derivative of shape (#batches, len/pooling_shape[0], width/pooling_shape[1], #channels/pooling_shape[2])
        """
        ar = np.ones(inputs.shape, dtype=np.float)
        return np.divide(ar, self.poolingShape[0] * self.poolingShape[1] * self.poolingShape[2])

    def output(self, inputs, bias=None, **kwargs):
        """
        Activate the layer
        :@param inputs: ndarray of shape (#batches, len, width, #channels)  #samples are equal to #batches
        :@param bias: ndarray of nchannels length, containing 0 or 1; or None if no bias
        :@return: output ndarray of shape (#batches, l/pooling_shape[0], w/pooling_shape[1], nchannels/pooling_shape[2])
        """
        assert len(inputs.shape) == 4
        dims = [inputs.shape[1], inputs.shape[2], inputs.shape[3]]
        assert all([i % p == 0 for i, p in zip(dims, self.poolingShape)])

        output_shape = (inputs.shape[0], inputs.shape[1]//self.poolingShape[0], inputs.shape[2]//self.poolingShape[1],
                        inputs.shape[3]//self.poolingShape[2])
        pool_shape = self.poolingShape
        rows_list = [range(i * pool_shape[0], (i+1) * pool_shape[0]) for i in range(output_shape[1])]
        cols_list = [range(i * pool_shape[1], (i+1) * pool_shape[1]) for i in range(output_shape[2])]
        dep_list = [range(i * pool_shape[2], (i+1) * pool_shape[2]) for i in range(output_shape[3])]
        output = self.POOL_FUNC[self.poolingType](inputs[:, rows_list, :, :][:, :, :, cols_list, :][:, :, :, :, :, dep_list], axis=(2, 4, 6))
        return output

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation algorithm for the layer
        @param inputs: numpy ndarray (4 dimensions): #batches X #len X #width X #depth
        @param layer_deriv: derivative of activation function for each neuron: d(activation_func).
        4 dimensional: #batches X #input_len X #input_width X #channels
        @param delta_arr_layer: 5 dimensional ndarray containing dy_{network_output_neuron}/dy_j.
        dimensions: (#batches, #network output neurons, #len, #width, #channels)
        This must be provided
        @param deriv_loss_wrt_outputs: dLoss/dy_{network_output_neurons} : 2 dimensional ndarray: (#batches, #output_neurons_in_network)
        @param bias: Pooling layer does not use bias
        @return: empty ndarray (no trainable weights) and  delta_arr_lastlayer: This is calculated by backpropagation. It contains dy_{output_neuron}/dy_k for y_k outputs from
        previous layer that are inputs to this layer. Dimensions: (#batches, #output_neurons, input dimension from last layer)
        """
        assert len(layer_deriv.shape) == 4
        ar1 = np.repeat(delta_arr_layer, self.poolingShape[0], axis=2)
        ar1 = np.repeat(ar1, self.poolingShape[1], axis=3)
        delta_arr_lastlayer = np.repeat(ar1, self.poolingShape[2], axis=4)
        return np.array([]), delta_arr_lastlayer

    def outputShape(self, input_shape=None):
        """
        Return the output shape of this layer
        @param input_shape: shape of input: tuple with 4 elements: (#batches, len, width, depth)
        @return: output shape tuple
        """
        if input_shape is None:
            return None, None, None, None
        return input_shape[0], input_shape[1] // self.poolingShape[0], input_shape[2] // self.poolingShape[1], input_shape[3] // self.poolingShape[2]

    def applyWeightsToInput(self, inputs, bias=None):
        return inputs

    def deriv(self, inputs, bias=None):
        return getattr(self, self.POOL_DERIV_FUNC[self.poolingType])(inputs)

    def inputShape(self):
        return None, None, None, None

    def applyWeightCorrections(self, weight_correction):
        pass

    def getWeights(self):
        pass

    def setWeights(self, weights):
        pass


class PoolingLayer2D(PoolingLayer3D):
    """
    2D Pooling layer for pooling the output of a convolution layer. Layer has no trainable weights.
    """

    def __init__(self, pooling_shape, pooling_type=PoolingType.MAX):
        """
        Initialize the pooling layer
        :@param pooling_shape: (pooled_len, pooled_width) tuple
        :@param pooling_type: PoolingType
        """
        assert isinstance(pooling_type, PoolingType)
        assert len(pooling_shape) == 2
        super(PoolingLayer2D, self).__init__(pooling_shape + (1,), pooling_type)

    def maxDeriv(self, inputs):
        """
        Derivative of max pooling layer
        :@param inputs: ndarray of (#batches, len, width, #channels) shape
        :@return: derivative of shape (#batches, len, width, #channels)
        """
        output_shape = (inputs.shape[0],
                        inputs.shape[1] // self.poolingShape[0],
                        inputs.shape[2] // self.poolingShape[1],
                        inputs.shape[3])
        pool_shape = self.poolingShape
        rows_list = [range(i * pool_shape[0], (i + 1) * pool_shape[0]) for i in range(output_shape[1])]
        cols_list = [range(i * pool_shape[1], (i + 1) * pool_shape[1]) for i in range(output_shape[2])]
        max_val = np.max(inputs[:, rows_list, :, :][:, :, :, cols_list, :], axis=(2, 4))
        val = (inputs[:, rows_list, :, :][:, :, :, cols_list, :] == max_val[:, :, np.newaxis, :, np.newaxis, :])
        val = np.where(val, 1, 0)
        return val.reshape(inputs.shape)

    def meanDeriv(self, inputs):
        """
        Derivative of mean pooling layer
        :@param inputs: ndarray of (#batches, len, width, #channels) shape
        :@return: derivative of shape (#batches, len/pooling_shape[0], width/pooling_shape[1], #channels)
        """
        ar = np.ones(inputs.shape, dtype=np.float)
        return np.divide(ar, self.poolingShape[0] * self.poolingShape[1])

    def output(self, inputs, bias=None, **kwargs):
        """
        Activate the layer
        :@param inputs: ndarray of shape (#batches, len, width, #channels)  #samples are equal to #batches
        :@param bias: ndarray of nchannels length, containing 0 or 1; or None if no bias
        :@return: output ndarray of shape (#batches, l/pooling_shape[0], w/pooling_shape[1], nchannels)
        """
        assert len(inputs.shape) == 4
        dims = [inputs.shape[1], inputs.shape[2]]
        assert all([i % p == 0 for i, p in zip(dims, self.poolingShape)])

        output_shape = (inputs.shape[0], inputs.shape[1] // self.poolingShape[0], inputs.shape[2] // self.poolingShape[1],
                        inputs.shape[3])
        pool_shape = self.poolingShape
        rows_list = [range(i * pool_shape[0], (i+1) * pool_shape[0]) for i in range(output_shape[1])]
        cols_list = [range(i * pool_shape[1], (i+1) * pool_shape[1]) for i in range(output_shape[2])]
        output = self.POOL_FUNC[self.poolingType](inputs[:, rows_list, :, :][:, :, :, cols_list, :], axis=(2, 4))
        return output

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation algorithm for the layer
        @param inputs: numpy ndarray (4 dimensions): #batches X #len X #width X #depth
        @param layer_deriv: derivative of activation function for each neuron: d(activation_func).
        4 dimensional: #batches X #input_len X #input_width X #channels
        @param delta_arr_layer: 5 dimensional ndarray containing dy_{network_output_neuron}/dy_j.
        dimensions: (#batches, #network output neurons, #len, #width, #channels)
        This must be provided
        @param deriv_loss_wrt_outputs: dLoss/dy_{network_output_neurons} : 2 dimensional ndarray: (#batches, #output_neurons_in_network)
        @param bias: Pooling layer does not use bias
        @return: empty ndarray (no trainable weights) and  delta_arr_lastlayer: This is calculated by backpropagation. It contains dy_{output_neuron}/dy_k for y_k outputs from
        previous layer that are inputs to this layer. Dimensions: (#batches, #output_neurons, input dimension from last layer)
        """
        assert len(layer_deriv.shape) == 4
        ar1 = np.repeat(delta_arr_layer, self.poolingShape[0], axis=2)
        delta_arr_lastlayer = np.repeat(ar1, self.poolingShape[1], axis=3)
        return np.array([]), delta_arr_lastlayer


class ConvolutionLayer(Layer):
    """
    Convolution layer that applies an activation function to a weighted sum of inputs, applying same weights to
    different sections of the input, as specified by filter size, stride and padding
    """

    def __init__(self, filter_shape, input_depth, num_channels, stride=(1,1), padding=(0,0),
                 activation_func=UnitActivation(), init_wt=1E-2):
        """
        Initialize the convolution layer
        @param filter_shape: Tuple specifying (length, width) of the filter
        @param input_depth: input depth
        @param num_channels: Number of channels
        @param stride: Stride (int, int) specifying the stride to apply along length and width
        @param padding: Tuple (int, int) specifying padding to be applied along length and width
        @param activation_func: Activation function. Must be an object of type Activation
        @param init_wt: initial weight multiplier, initial weights between 0 and init_wt
        """
        assert isinstance(activation_func, Activation)
        self.filterShape = filter_shape
        self.nChannels = num_channels
        self.stride = stride
        self.padding = padding
        self.depth = input_depth
        self.weights = np.random.random((self.filterShape[0], self.filterShape[1], input_depth, self.nChannels)) * init_wt
        self.biasWeights = np.random.random(self.nChannels) * init_wt
        self.activationFunc = activation_func

    def paddedInput(self, inputs):
        """
        Apply padding to input
        @param inputs: unpadded input
        @return: padded input
        """
        if (self.padding[0] != 0) or (self.padding[1] != 0):
            xinp = np.zeros((inputs.shape[0],
                             inputs.shape[1] + 2*self.padding[0],
                             inputs.shape[2] + 2*self.padding[1],
                             inputs.shape[3]), dtype=np.float)
            xinp[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], :] = inputs
            inputs = xinp
        return inputs

    def applyWeightsToInput(self, padded_inputs, bias=None):
        """
        Apply weights to the layer to get an output before applying activation function
        :param padded_inputs: ndarray of shape (#batches, len, width, height)
                              If padding is applicable, this MUST be padded input
        :param bias: ndarray of shape (#batches, nchannels), containing 0 or 1; or None if no bias
        :return: output ndarray of shape (#batches, l, w, nchannels) where
        l = (padded_inputs.shape[1] - self.filterShape[0])/self.stride[0] + 1
        w = (padded_inputs.shape[2] - self.filterShape[1])/self.stride[1] + 1
        """
        assert len(padded_inputs.shape) == 4
        assert (padded_inputs.shape[1] - self.filterShape[0]) % self.stride[0] == 0
        assert (padded_inputs.shape[2] - self.filterShape[1]) % self.stride[1] == 0
        output_shape = (padded_inputs.shape[0],
                        (padded_inputs.shape[1] - self.filterShape[0]) // self.stride[0] + 1,
                        (padded_inputs.shape[2] - self.filterShape[1]) // self.stride[1] + 1,
                        self.nChannels)
        if bias is not None:
            assert bias.shape == (padded_inputs.shape[0], self.nChannels)

        rows_list = [range(i * self.stride[0], i * self.stride[0] + self.filterShape[0]) for i in range(output_shape[1])]
        cols_list = [range(j * self.stride[1], j * self.stride[1] + self.filterShape[1]) for j in range(output_shape[2])]
        output = np.einsum("ijklmn,kmno->ijlo", padded_inputs[:, rows_list, :, :][:, :, :, cols_list, :], self.weights)
        if bias:
            output += np.einsum("ij,j->ij", bias, self.biasWeights)[:, np.newaxis, np.newaxis, :]

        return output

    def output(self, padded_inputs, bias=None, **kwargs):
        """
        Activate the layer
        :param padded_inputs: ndarray of shape (#batches, len, width, height)
                              If padding is applicable, this MUST be padded input
        :param bias: ndarray of nchannels length, containing 0 or 1; or None if no bias
        :return: output ndarray of shape (#batches, l, w, nchannels) where
        l = (padded_inputs.shape[1] - self.filterShape[0])/self.stride[0] + 1
        w = (padded_inputs.shape[2] - self.filterShape[1])/self.stride[1] + 1
        """
        output = self.applyWeightsToInput(padded_inputs, bias)
        return self.activationFunc.output(output)

    def deriv(self, padded_inputs, bias=None):
        """ Calculate dyi/dai """
        assert len(padded_inputs.shape) == 4
        weighted_input = self.applyWeightsToInput(padded_inputs, bias)
        return self.activationFunc.deriv(weighted_input)

    def backPropagation(self, padded_inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation algorithm for the layer
        @param padded_inputs: numpy ndarray (4 dimensions): #batches X len X width X depth
        @param layer_deriv: derivative of activation function for each neuron.
                            4 dimensional ndarray of shape (#batches, output len, output width, #channnels)
        @param delta_arr_layer: 5 dimensional ndarray containing dy_{output_neuron}/dy_j for neurons from this layer
        Dimensions: (#batches, #network output neurons, layer_output_len, layer_output_width, channels)
        @param deriv_loss_wrt_outputs: dLoss/dy_{output_neurons} : 2 dimensional ndarray: (#batches, #output_neurons_in_network)
        @param bias: optional. 0 if not specified. Else, 2 dimensional array of shape (#batches, #channels)
        @return: ndarray with dLoss/dweight_{i,j} and delta_arr_lastlayer
        """
        output_len = (padded_inputs.shape[1] - self.filterShape[0]) // self.stride[0] + 1
        output_width = (padded_inputs.shape[2] - self.filterShape[1]) // self.stride[1] + 1
        input_len = padded_inputs.shape[1] - 2*self.padding[0]
        input_width = padded_inputs.shape[2] - 2*self.padding[1]
        input_depth = padded_inputs.shape[3]
        delta_arr_lastlayer = np.zeros((delta_arr_layer.shape[0], delta_arr_layer.shape[1], input_len, input_width, input_depth), dtype=np.float)
        bias_weight_gradients = np.zeros(self.biasWeights.shape, dtype=np.float)

        extents = np.zeros(4, dtype=np.int)
        weight_gradients = np.zeros(self.weights.shape, dtype=np.float)
        for i in range(output_len):
            extents[0] = i * self.stride[0]
            extents[1] = extents[0] + self.filterShape[0]
            for j in range(output_width):
                extents[2] = j * self.stride[1]
                extents[3] = extents[2] + self.filterShape[1]
                delta_arr_lastlayer[:, :, extents[0]:extents[1], extents[2]:extents[3], :] += np.einsum("ijn,in,klmn->ijklm",
                                                                                                        delta_arr_layer[:, :, i, j, :],
                                                                                                        layer_deriv[:, i, j, :],
                                                                                                        self.weights)
                # calculate weight gradients
                weight_gradients += np.einsum("ij,ijn,in,iklm->klmn",
                                              deriv_loss_wrt_outputs,
                                              delta_arr_layer[:, :, i, j, :],
                                              layer_deriv[:, i, j, :],
                                              padded_inputs[:, extents[0]:extents[1], extents[2]:extents[3], :])

        if bias:
            bias_weight_gradients = np.einsum("ij,ijkln,ikln,in->n",
                                              deriv_loss_wrt_outputs,
                                              delta_arr_layer,
                                              layer_deriv,
                                              bias)
        return (weight_gradients, bias_weight_gradients), delta_arr_lastlayer

    def inputShape(self):
        """
        Return shape of input
        @return: 4 element tuple. First 3 elements are input specific
        """
        return None, None, None, self.depth

    def outputShape(self, input_shape=None):
        """
        Return output shape for output generated by this layer
        @param input_shape: input shape (padded input)
        @return: 4 element tuple: (#batches, output len, output width, #channels)
        """
        output_len = (input_shape[1] - self.filterShape[0]) / self.stride[0] + 1
        output_width = (input_shape[2] - self.filterShape[1]) / self.stride[1] + 1
        return input_shape[0], output_len, output_width, self.nChannels

    def applyWeightCorrections(self, weight_correction):
        """
        Apply weight corrections
        @param weight_correction: List with first element as weight corrections and second as bias weight corrections
        """
        self.weights = np.add(self.weights, weight_correction[0])
        self.biasWeights = np.add(self.biasWeights, weight_correction[1])

    def getWeights(self):
        """
        Get weights. Returns a reference, changing the returned value will directly change the layer weights.
        @return: Reference to layer weights
        """
        return self.weights, self.biasWeights

    def setWeights(self, weights):
        """
        Set layer weights
        @param weights: Tuple or list with 2 elements: weights and bias weights), same shape as weights and bias weights respectively
        """
        self.weights[:, :, :, :] = weights[0]
        self.biasWeights[:] = weights[1]


class FlattenLayer(Layer):
    """ Flatten a d-dimensional layer to a 1 dimensional layer. This layer has no trainable weights. """
    def output(self, inputs, bias=None, **kwargs):
        """
        Flatten the output
        @param inputs: d dimensional array: (#batches, remaining dimensions)
        @param bias:
        @return: Flattened array: (#batches, 1dimension)
        """
        return inputs.reshape((inputs.shape[0], -1))

    def applyWeightsToInput(self, inputs, bias=None):
        return inputs.reshape((inputs.shape[0], -1))

    def deriv(self, inputs, bias=None, activ_val=None):
        flat_dim = functools.reduce(lambda a, b: a * b, inputs.shape[1:])
        return np.ones((inputs.shape[0], flat_dim), dtype=np.bool)

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        new_shape = (delta_arr_layer.shape[0], delta_arr_layer.shape[1]) + inputs.shape[1:]
        delta_arr_lastlayer = delta_arr_layer.reshape(new_shape)
        return np.array([]), delta_arr_lastlayer

    def inputShape(self):
        """
        Does not track the shape of input, get from the previous layer
        @return: None
        """
        return None

    def outputShape(self, input_shape=None):
        return input_shape[0], functools.reduce(lambda a, b: a*b, input_shape[1:])

    def applyWeightCorrections(self, weight_correction):
        pass

    def getWeights(self):
        pass

    def setWeights(self, weights):
        pass


class SoftmaxLayer(Layer):
    """
    Softmax layer: convert to probability distribution across discrete buckets.
    This layer has no trainable parameters.
     """
    def __init__(self, buckets=None):
        self.nBuckets = buckets

    def output(self, inputs, bias=None, **kwargs):
        """
        Output of softmax layer
        @param inputs: 2 dimensional ndarray of shape (#batches, inputs)
        @param bias: None (no bias)
        """
        self.nBuckets = inputs.shape[1]
        max_val = inputs.max(axis=1)
        exp_vals = np.exp(np.subtract(inputs, max_val[:, np.newaxis]))
        sum_vals = exp_vals.sum(axis=1)
        return np.divide(exp_vals, sum_vals[:, np.newaxis])

    def applyWeightsToInput(self, inputs, bias=None):
        return inputs

    def deriv(self, inputs, bias=None, activ_val=None):
        """
        Derivative dy_i/dx_j . Matrix of shape (#batches, #buckets, #buckets)
        @param inputs: 2 dimensional matrix of shape (#batches, #buckets)
        @param bias: None
        @param activ_val: optional. output of this layer
        @return: derivative dy_i/dx_j
        """

        if activ_val is None:
            activ_val = self.output(inputs, bias)
        identity_matrix = np.eye(inputs.shape[1], dtype=np.float)
        diag = np.multiply(activ_val[:, :, np.newaxis], identity_matrix[np.newaxis, :, :])
        yiyj = np.einsum("ij,ik->ijk", activ_val, activ_val)
        return np.subtract(diag, yiyj)

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Backpropagate gradients. This layer has no trainable paramaters, so just needs to calculate delta_arr_lastlayer
        @param inputs: ndarray of shape (#batches, #buckets)
        @param layer_deriv: ndarray of shape (#batches, #buckets, #buckets)
        @param delta_arr_layer: ndarray of shape (#batches, #buckets)
        @param deriv_loss_wrt_outputs: 2d ndarray of shape (#batches, #network output neurons)
        @param bias: None. Softmax layer has no bias
        @return: Tuple: (empty array, dy_output/dy_previous_layer)
        """
        delta_arr_lastlayer = np.einsum("ijk,ikl->ijl", delta_arr_layer[:, 0:-1, :], layer_deriv)
        return np.array([]), delta_arr_lastlayer

    def inputShape(self):
        return None, self.nBuckets

    def outputShape(self, input_shape=None):
        return None, self.nBuckets

    def applyWeightCorrections(self, weight_correction):
        pass

    def getWeights(self):
        pass

    def setWeights(self, weights):
        pass


class DropoutLayer(Layer):
    """ Dropout layer. Randomly sets a fraction of inputs to 0 during training. Has no impact during prediction.
     Layer has no trainable weights """

    def __init__(self, dropout_fraction):
        self.f = dropout_fraction
        self.TRAINING = True

    def output(self, inputs, bias=None, **kwargs):
        if self.TRAINING:
            mult_factor = self.applyWeightsToInput(inputs, bias)
            return np.multiply(inputs, mult_factor)
        return inputs

    def deriv(self, inputs, bias=None):
        return self.applyWeightsToInput(inputs, bias)

    def applyWeightsToInput(self, inputs, bias=None):
        len_inputs = functools.reduce(lambda x, y=1: x * y, inputs.shape[1:])
        set_to_zero = int(self.f * len_inputs)
        zero_elems = np.random.choice(len_inputs, set_to_zero)
        filter = np.ones(inputs.shape[1:], dtype=np.float)
        filter.reshape(-1)[zero_elems] = 0.0
        mult_factor = len_inputs/float(len_inputs - set_to_zero)
        return np.multiply(filter[np.newaxis, :], mult_factor)

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        factor = self.applyWeightsToInput(inputs)
        delta_arr_lastlayer = np.multiply(delta_arr_layer, factor)
        return np.array([]), delta_arr_lastlayer

    def inputShape(self):
        return None

    def outputShape(self, input_shape=None):
        return input_shape

    def applyWeightCorrections(self, weight_correction):
        pass

    def getWeights(self):
        pass

    def setWeights(self, weights):
        pass


class EmbeddingLayer(Layer):
    """ Embedding layer that maps an input into dense embedding """

    def __init__(self, vocab_size, embedding_size, one_hot_input=False):
        """
        Initialize the layer
        @param vocab_size: Size of vocabulary.
        @param embedding_size: Size of embedding
        @param one_hot_input: True if inputs have one hot encoding. False if they are indices between 0 and vocab_size-1 (inclusive)
        """
        self.vocabSize = vocab_size
        self.embeddingSize = embedding_size
        self.oneHotInput = one_hot_input
        self.weights = np.random.random((vocab_size, embedding_size))

    def output(self, inputs, bias=None, **kwargs):
        """
        Output of the layer
        @param inputs: If one hot encoding is used, 3 dimensional ndarray of shape (#batches, #input size, vocabulary size)
        If one hot encoding is False, 2 dimensional ndarray of shape (#batches, #input size)
        @param bias: Not used
        @return: Output of the layer containing embedding, ndarray of shape (#batches, #input size, embedding size)
        """
        one_hot_inputs = inputs
        if not self.oneHotInput:
            # convert inputs to one hot encoded inputs
            one_hot_inputs = np.zeros((inputs.shape[0], inputs.shape[1], self.vocabSize), dtype=np.bool)
            one_hot_inputs[np.arange(inputs.shape[0])[:, np.newaxis, np.newaxis],
                           np.arange(inputs.shape[1])[:, np.newaxis],
                           inputs[:, :, np.newaxis]] = True

        return np.einsum("ijk,kl->ijl", one_hot_inputs, self.weights)

    def deriv(self, inputs, bias=None):
        return np.ones((inputs.shape[0], inputs.shape[1], self.embeddingSize), dtype=np.bool)

    def applyWeightsToInput(self, inputs, bias=None):
        return self.output(inputs, bias)

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        # layer deriv contains all 1
        delta_arr_lastlayer = np.einsum("ijk,lk->ijl", delta_arr_layer, self.weights)
        # calculate weight gradients
        weight_gradients = np.einsum("ij,ijk,ilm->km", deriv_loss_wrt_outputs, delta_arr_layer, inputs)
        return weight_gradients, delta_arr_lastlayer

    def inputShape(self):
        if self.oneHotInput:
            return None, None, self.vocabSize
        return None, None

    def outputShape(self, input_shape=None):
        return None, None, self.embeddingSize

    def applyWeightCorrections(self, weight_correction):
        self.weights = np.add(self.weights, weight_correction)

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights[:, :] = weights


class GRULayer(Layer):
    """
    A layer containing GRU (Gated Recurrence Unit) cells.
    Cell output is cell state
    X_t: input, h_t-1, previous cell state, Z_t: update gate, hpred_t: candidate cell state, r_t: reset gate
    Z_t = sigmoid(W_z * X_t + U_z * h_t-1 + b_z
    r_t = sigmoid(W_r * X_t + U_r * h_t-1 + b_r
    hpred_t = tanh(W_h * X_t + U_h * r_t * h_t-1 + b_h
    h_t = (1 - Z_t)*h_t-1 + Z_t * hpred_t
    """

    def __init__(self, num_units, num_features, update_gate_activation=Sigmoid(),
                 reset_gate_activation=Sigmoid(), output_activation=HyperbolicTangent(),
                 init_wt=1E-2, init_state=0):
        """
        Initialize the GRU layer
        @param num_units: Number of GRU units
        @param num_features: Number of features in input
        @param update_gate_activation:
        @param reset_gate_activation:
        @param output_activation:
        @param init_wt: initial weight are are random numbers in range [0, init_wt]
        @param init_state: initial state of the cell
        """
        self.nUnits = num_units
        self.nFeatures = num_features
        assert isinstance(update_gate_activation, Activation)
        assert isinstance(reset_gate_activation, Activation)
        assert isinstance(update_gate_activation, Activation)
        self.updateGateActiv = update_gate_activation
        self.resetGateActiv = reset_gate_activation
        self.outputGateActiv = output_activation
        self.updateGateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.resetGateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.outputGateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.initState = init_state

    def setInitialState(self, state):
        self.initState = state

    def output(self, inputs, bias=None, **kwargs):
        """
        Output of the layer
        @param inputs: 3 dimensional ndarray of shape (#batches, #timesteps, #features)
        @param bias: None (0 bias) or 4 dimensional ndarray of shape (#batches, #timesteps, #units, gate number)
        gate number = 0 for update gate
        gate number = 1 for reset gate
        gate number = 2 for output gate
        @param kwargs: return_seq=True to return output for all timesteps,
        derivs=True to return derivatives,
        put_initial_state=True for initial state at t=0 to be added to output as first element. This means that
        output[:, i, :] will correspond to output for time step i-1
        @return: Tuple of all_outputs and derivatives. Depending upon kwargs, output will be a 3 dimensional ndarray
        of shape (#batches, #timesteps, #units) if return_sequences=True, or 2d ndarray of shape (#batches, #units) if
        return_sequences=False
        Derivative (if requested) will be a tuple with 3 3d ndarrays each of shape (#batches, #timesteps, #units)
        The 3 arrays correspond to the 3 gates: update gate, reset gate and output gate
        """
        update_gate_inp = np.einsum("ijk,kl->ijl", inputs, self.updateGateWeights[0:-2, :])
        reset_gate_inp = np.einsum("ijk,kl->ijl", inputs, self.resetGateWeights[0:-2, :])
        output_gate_inp = np.einsum("ijk,kl->ijl", inputs, self.outputGateWeights[0:-2, :])
        if bias:
            update_gate_inp = update_gate_inp + np.einsum("ijk,k->ijk", bias[:, :, :, 0], self.updateGateWeights[-1, :])
            reset_gate_inp = reset_gate_inp + np.einsum("ijk,k->ijk", bias[:, :, :, 1], self.resetGateWeights[-1, :])
            output_gate_inp = output_gate_inp + np.einsum("ijk,k->ijk", bias[:, :, :, 2], self.outputGateWeights[-1, :])

        state = np.ndarray((inputs.shape[0], self.nUnits), dtype=np.float)
        state[:, :] = self.initState
        return_seq = kwargs.get("return_seq", False)
        put_initial_state = kwargs.get("put_initial_state", False)
        all_outputs, ug_outputs, rg_outputs, og_outputs, begin = None, None, None, None, None
        if return_seq:
            if put_initial_state:
                all_outputs = np.zeros((inputs.shape[0], inputs.shape[1] + 1, self.nUnits), dtype=np.float)
                all_outputs[:, 0, :] = state
                begin = 1
            else:
                all_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
                begin = 0
            ug_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            rg_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            og_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)

        derivs_required = kwargs.get("derivs", False)
        zt, rt, hpredt = None, None, None
        update_gate_derivs, reset_gate_derivs, output_gate_derivs = None, None, None
        if derivs_required:
            update_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            reset_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            output_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)

        for i in range(inputs.shape[1]):
            upd_input = update_gate_inp[:, i, :] + np.einsum("ij,j->ij", state, self.updateGateWeights[-2, :])
            reset_input = reset_gate_inp[:, i, :] + np.einsum("ij,j->ij", state, self.resetGateWeights[-2, :])
            zt = self.updateGateActiv.output(upd_input)
            rt = self.resetGateActiv.output(reset_input)
            output_inp = output_gate_inp[:, i, :] + np.einsum("ij,ij,j->ij", rt, state, self.outputGateWeights[-2, :])
            hpredt = self.outputGateActiv.output(output_inp)
            state = np.multiply(1 - zt, state) + np.multiply(zt, hpredt)
            if return_seq:
                all_outputs[:, begin + i, :] = state
                ug_outputs[:, i, :] = zt
                rg_outputs[:, i, :] = rt
                og_outputs[:, i, :] = hpredt
            if derivs_required:
                update_gate_derivs[:, i, :] = self.updateGateActiv.deriv(upd_input)
                reset_gate_derivs[:, i, :] = self.resetGateActiv.deriv(reset_input)
                output_gate_derivs[:, i, :] = self.outputGateActiv.deriv(output_inp)

        if return_seq:
            return (all_outputs, ug_outputs, rg_outputs, og_outputs), (update_gate_derivs, reset_gate_derivs, output_gate_derivs)
        return (state, zt, rt, hpredt), (update_gate_derivs, reset_gate_derivs, output_gate_derivs)

    def deriv(self, inputs, bias=None):
        return self.output(inputs, bias, derivs=True)[1]

    def applyWeightsToInput(self, inputs, bias=None):
        pass

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation though time
        @param inputs: 3d ndarray of shape (#batches, #timesteps, #features)
        @param layer_deriv: Not used
        @param delta_arr_layer: dy_network_output_neuron/dy_layer
        This is a 3d ndarray of shape (#batches, #network output neurons, #units)
        @param deriv_loss_wrt_outputs: dLoss/dy_layer. 2d ndarray of shape (#batches, #units)
        @param bias: None for no bias. Else a 4 dimensional ndarray of shape (#batches, #timesteps, #units, gate number)
        gate number = 0 for update gate
        gate number = 1 for reset gate
        gate number = 2 for output gate
        @return Tuple with 2 elements:
        First element is a tuple with weight gradients, one for each gate
        Second element is dy_network_output_neuron/dy_last_layer, 3d ndarray of shape (#batches, #network output neurons, #units)
        """
        outputs, derivs = self.output(inputs, bias, derivs=True, return_seq=True, put_initial_state=True)
        output, ug_output, rg_output, og_output = outputs
        ug_derivs, rg_derivs, og_derivs = derivs
        ug_weight_gradients = np.zeros(self.updateGateWeights.shape, dtype=np.float)
        rg_weight_gradients = np.zeros(self.resetGateWeights.shape, dtype=np.float)
        og_weight_gradients = np.zeros(self.outputGateWeights.shape, dtype=np.float)

        delta_curr_layer = delta_arr_layer
        for i in range(inputs.shape[1] - 1, -1, -1):
            # calculate update gate weight gradients
            common = np.einsum("ij,ijk->ik", deriv_loss_wrt_outputs, delta_curr_layer)
            diff_matrix = np.subtract(og_output[:, i, :], output[:, i, :]) # output corresponds to previous time step because put_initial_state=True
            matrix = np.einsum("ik,ik,ik->ik", common,
                               ug_derivs[:, i, :],
                               diff_matrix)
            ug_weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            ug_weight_gradients[-2, :] += np.einsum("ik,ik->k", matrix, output[:, i, :])
            if bias:
                ug_weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :, 0])

            # calculate reset gate weight gradients
            matrix = np.einsum("ik,ik,ik,k,ik,ik->ik", common,
                               ug_output[:, i, :],
                               og_derivs[:, i, :],
                               self.outputGateWeights[-2, :],
                               rg_derivs[:, i, :],
                               output[:, i, :])
            rg_weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            rg_weight_gradients[-2, :] += np.einsum("ik,ik->k", matrix, output[:, i, :])
            if bias:
                rg_weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :, 1])

            # calculate output gate weight gradients
            matrix = np.einsum("ik,ik,ik->ik", common,
                               og_derivs[:, i, :],
                               ug_output[:, i, :])
            og_weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            og_weight_gradients[-2, :] += np.einsum("ik,ik,ik->k", matrix, rg_output[:, i, :], output[:, i, :])
            if bias:
                og_weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :, 2])

            # calculate derivatives dy_l/dy_j for last layer, y_l : output, y_j: last layer's output
            matrix = np.subtract(1, ug_output[:, i, :])
            matrix -= np.einsum("ij,j,ij->ij", ug_derivs[:, i, :], self.outputGateWeights[-2, :], output[:, i, :])
            matrix += np.einsum("ij,j,ij->ij", ug_derivs[:, i, :], self.outputGateWeights[-2, :], og_output[:, i, :])
            matrix += np.einsum("ij,ij,j,ij,ij,j->ij", ug_output[:, i, :], og_derivs[:, i, :], self.outputGateWeights[-2, :],
                                rg_derivs[:, i, :], output[:, i, :], self.resetGateWeights[-2, :])
            delta_curr_layer[:, :, :] = np.einsum("ijk,ik->ijk", delta_curr_layer, matrix)

        ug_weight_gradients[:, :] = np.divide(ug_weight_gradients, inputs.shape[1])
        rg_weight_gradients[:, :] = np.divide(rg_weight_gradients, inputs.shape[1])
        og_weight_gradients[:, :] = np.divide(og_weight_gradients, inputs.shape[1])
        return (ug_weight_gradients, rg_weight_gradients, og_weight_gradients), delta_curr_layer

    def inputShape(self):
        return None, None, self.nFeatures

    def outputShape(self, input_shape=None):
        return None, None, self.nUnits

    def applyWeightCorrections(self, weight_correction):
        """
        Apply weight corrections
        @param weight_correction: tuple or list with 3 ndarrays containing corrections for weights of 3 gates in this cell
        """
        self.updateGateWeights[:, :] = np.add(self.updateGateWeights, weight_correction[0])
        self.resetGateWeights[:, :] = np.add(self.resetGateWeights, weight_correction[1])
        self.outputGateWeights[:, :] = np.add(self.outputGateWeights, weight_correction[2])

    def getWeights(self):
        return self.updateGateWeights, self.resetGateWeights, self.outputGateWeights

    def setWeights(self, weights):
        self.updateGateWeights[:, :] = weights[0]
        self.resetGateWeights[:, :] = weights[1]
        self.outputGateWeights[:, :] = weights[2]


class LSTMLayer(Layer):
    """
    Layer comprising of LSTM (Long Short Term Memory) cells
    forget gate: f_t = sigmoid(Wf * X_t + Uf * h_t-1 + bf)
    input gate: i_t = sigmoid(Wi * X_t + Ui * h_t-1 + bi)
    output gate: o_t = sigmoid(Wo * X_t + Uo * h_t-1 + bo)
    predicted state: cpred = tanh(Wc * X_t + Uc * h_t-1 + bc)
    cell state: c_t = f_t * c_t-1 + i_t * cpred
    hidden state: ht = o_t * tanh(c_t)
    Output of the cell at each step is ht, i.e. the hidden state
    """

    def __init__(self, num_units, num_features, forget_gate_activation=Sigmoid(),
                 input_gate_activation=Sigmoid(), output_gate_activation=Sigmoid(),
                 cell_state_activation=HyperbolicTangent(), hidden_state_activation=HyperbolicTangent(),
                 init_wt=1E-2, init_cell_state=0, init_hidden_state=0):
        """
        Initialize the LSTM layer
        @param num_units:
        @param num_features:
        @param forget_gate_activation:
        @param input_gate_activation:
        @param output_gate_activation:
        @param cell_state_activation:
        @param hidden_state_activation:
        @param init_wt:
        @param init_cell_state:
        @param init_hidden_state:
        """
        self.nUnits = num_units
        self.nFeatures = num_features
        assert isinstance(forget_gate_activation, Activation)
        assert isinstance(input_gate_activation, Activation)
        assert isinstance(output_gate_activation, Activation)
        assert isinstance(cell_state_activation, Activation)
        assert isinstance(hidden_state_activation, Activation)
        self.forgetGateActiv = forget_gate_activation
        self.inputGateActiv = input_gate_activation
        self.outputGateActiv = output_gate_activation
        self.cellStateActiv = cell_state_activation
        self.hiddenStateActiv = hidden_state_activation
        self.forgetGateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.inputGateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.outputGateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.cellStateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.initCellState = init_cell_state
        self.initHiddenState = init_hidden_state

    def output(self, inputs, bias=None, **kwargs):
        """
        Output of the layer
        @param inputs: 3 dimensional ndarray of shape (#batches, #timesteps, #features)
        @param bias: None (0 bias) or 4 dimensional ndarray of shape (#batches, #timesteps, #units, gate number)
        gate number = 0 for forget gate
        gate number = 1 for input gate
        gate number = 2 for output gate
        gate number = 3 for cell state gate
        @param kwargs: return_seq=True to return output for all timesteps,
        derivs=True to return derivatives,
        put_initial_state=True for initial state at t=0 to be added to output as first element. This means that
        output[:, i, :] will correspond to output for time step i-1
        @return: Tuple of all_outputs and derivatives. Depending upon kwargs, output will be a 3 dimensional ndarray
        of shape (#batches, #timesteps, #units) if return_sequences=True, or 2d ndarray of shape (#batches, #units) if
        return_sequences=False
        Derivative (if requested) will be a tuple with 4 3d ndarrays each of shape (#batches, #timesteps, #units)
        The 4 arrays correspond to the 4 gates
        """
        forget_gate_inp = np.einsum("ijk,kl->ijl", inputs, self.forgetGateWeights[0:-2, :])
        input_gate_inp = np.einsum("ijk,kl->ijl", inputs, self.inputGateWeights[0:-2, :])
        output_gate_inp = np.einsum("ijk,kl->ijl", inputs, self.outputGateWeights[0:-2, :])
        cell_state_inp = np.einsum("ijk,kl->ijl", inputs, self.cellStateWeights[0:-2, :])
        if bias:
            forget_gate_inp = forget_gate_inp + np.einsum("ijk,k->ijk", bias[:, :, :, 0], self.forgetGateWeights[-1, :])
            input_gate_inp = input_gate_inp + np.einsum("ijk,k->ijk", bias[:, :, :, 1], self.inputGateWeights[-1, :])
            output_gate_inp = output_gate_inp + np.einsum("ijk,k->ijk", bias[:, :, :, 2], self.outputGateWeights[-1, :])
            cell_state_inp = cell_state_inp + np.einsum("ijk,k->ijk", bias[:, :, :, 3], self.cellStateWeights[-1, :])

        cell_state = np.ndarray((inputs.shape[0], self.nUnits), dtype=np.float)
        cell_state[:, :] = self.initCellState
        hidden_state = np.ndarray((inputs.shape[0], self.nUnits), dtype=np.float)
        hidden_state[:, :] = self.initHiddenState
        ft, it, ot, cpred, cfinal, activt = None, None, None, None, None, None
        all_outputs, fg_outputs, ig_outputs, og_outputs, cpred_outputs, activt_outputs, begin = None, None, None, None, None, None, None
        return_seq = kwargs.get("return_seq", False)
        put_initial_state = kwargs.get("put_initial_state", False)
        if return_seq:
            if put_initial_state:
                all_outputs = np.zeros((inputs.shape[0], inputs.shape[1] + 1, self.nUnits), dtype=np.float)
                cfinal = np.zeros((inputs.shape[0], inputs.shape[1] + 1, self.nUnits), dtype=np.float)
                all_outputs[:, 0, :] = hidden_state
                cfinal[:, 0, :] = cell_state
                begin = 1
            else:
                all_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
                cfinal = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
                begin = 0
            fg_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            ig_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            og_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            cpred_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            activt_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)

        derivs_required = kwargs.get("derivs", False)
        forget_gate_derivs, input_gate_derivs, output_gate_derivs, cell_gate_derivs, ht_derivs = None, None, None, None, None
        if derivs_required:
            forget_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            input_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            output_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            cell_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
            ht_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)

        for i in range(inputs.shape[0]):
            forget_input = forget_gate_inp[:, i, :] + np.einsum("ij,j->ij", hidden_state, self.forgetGateWeights[-2, :])
            inp_input = input_gate_inp[:, i, :] + np.einsum("ij,j->ij", hidden_state, self.inputGateWeights[-2, :])
            out_input = output_gate_inp[:, i, :] + np.einsum("ij,j->ij", hidden_state, self.outputGateWeights[-2, :])
            cell_input = cell_state_inp[:, i, :] + np.einsum("ij,j->ij", hidden_state, self.cellStateWeights[-2, :])

            ft = self.forgetGateActiv.output(forget_input)
            it = self.inputGateActiv.output(inp_input)
            ot = self.outputGateActiv.output(out_input)
            cpred = self.cellStateActiv.output(cell_input)
            activt = self.hiddenStateActiv.output(cell_state)
            cell_state = np.multiply(ft, cell_state) + np.multiply(it, cpred)
            hidden_state = np.multiply(ot, activt)
            if return_seq:
                all_outputs[:, begin + i, :] = hidden_state
                fg_outputs[:, i, :] = ft
                ig_outputs[:, i, :] = it
                og_outputs[:, i, :] = ot
                cpred_outputs[:, i, :] = cpred
                activt_outputs[:, i, :] = activt
                cfinal[:, begin + i, :] = cell_state
            if derivs_required:
                forget_gate_derivs[:, i, :] = self.forgetGateActiv.deriv(forget_input)
                input_gate_derivs[:, i, :] = self.inputGateActiv.deriv(inp_input)
                output_gate_derivs[:, i, :] = self.outputGateActiv.deriv(out_input)
                cell_gate_derivs[:, i, :] = self.cellStateActiv.deriv(cell_input)
                ht_derivs[:, i, :] = self.hiddenStateActiv.deriv(cell_state)

        if return_seq:
            return (all_outputs, fg_outputs, ig_outputs, og_outputs, cpred_outputs, cfinal, activt_outputs), (forget_gate_derivs, input_gate_derivs, output_gate_derivs, cell_gate_derivs, ht_derivs)
        return (hidden_state, ft, it, ot, cpred, cell_state, activt), (forget_gate_derivs, input_gate_derivs, output_gate_derivs, cell_gate_derivs, ht_derivs)

    def deriv(self, inputs, bias=None):
        return self.output(inputs, bias, derivs=True)[1]

    def applyWeightsToInput(self, inputs, bias=None):
        pass

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation though time
        @param inputs: 3 dimensional ndarray of shape (#batches, #timesteps, #units)
        @param layer_deriv: (not used in BPTT)
        @param delta_arr_layer: (will change its contents) dy_network_output/dy_layer 3 d ndarray of shape (#batches, #network outputs, #units)
        @param deriv_loss_wrt_outputs: dLoss/dy_layer : 2 dimensional ndarray of shape (#batches, #units)
        @param bias: (optional) 0 if None. Else, 4 dimensional ndarray of shape (#batches, #timesteps, #units, gate_number)
        @return: tuple with 2 elements: derivative of weights and derivative of network output with respect to cell input
        """
        outputs, derivs = self.output(inputs, bias, derivs=True, return_seq=True, put_initial_state=True)
        output, fg_output, ig_output, og_output, cpred_output, cfinal, activt = outputs
        fg_derivs, ig_derivs, og_derivs, cs_derivs, ht_derivs = derivs
        fg_weight_gradients = np.zeros(self.forgetGateWeights.shape, dtype=np.float)
        ig_weight_gradients = np.zeros(self.inputGateWeights.shape, dtype=np.float)
        og_weight_gradients = np.zeros(self.outputGateWeights.shape, dtype=np.float)
        cs_weight_gradients = np.zeros(self.cellStateWeights.shape, dtype=np.float)
        delta_curr_layer = delta_arr_layer
        for i in range(inputs.shape[1]-1, -1, -1):
            # calculate forget gate weight gradients
            common = np.einsum("ij,ijk->ik", deriv_loss_wrt_outputs, delta_curr_layer)
            matrix = np.einsum("ik,ik,ik,ik,ik->ik", common,
                               og_output[:, i, :],
                               ht_derivs[:, i, :],
                               cfinal[:, i, :],
                               ht_derivs[:, i, :])
            fg_weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            fg_weight_gradients[-2, :] += np.einsum("ik,ik->k", matrix, output[:, i, :])
            if bias:
                fg_weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :, 0])

            # calculate input gate weight gradients
            matrix = np.einsum("ik,ik,ik,ik,ik->ik", common,
                               og_output[:, i, :],
                               ht_derivs[:, i, :],
                               cpred_output[:, i, :],
                               ig_derivs[:, i, :])
            ig_weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            ig_weight_gradients[-2, :] += np.einsum("ik,ik->k", matrix, output[:, i, :])
            if bias:
                ig_weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :, 1])

            # calculate output gate weight gradients
            matrix = np.einsum("ik,ik,ik->ik", common,
                               og_derivs[:, i, :],
                               activt[:, i, :])
            og_weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            og_weight_gradients[-2, :] += np.einsum("ik,ik->k", matrix, output[:, i, :])
            if bias:
                og_weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :, 2])

            # calculate cell state calculation cell's weight gradients
            matrix = np.einsum("ik,ik,ik,ik,ik->ik", common,
                               og_output[:, i, :],
                               ht_derivs[:, i, :],
                               ig_output[:, i, :],
                               cs_derivs[:, i, :])
            cs_weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            cs_weight_gradients[-2, :] += np.einsum("ik,ik->k", matrix, output[:, i, :])
            if bias:
                cs_weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :, 3])

            # calculate derivatives dy_l/dy_j for last layer, y_l : output, y_j: last layer's output
            matrix = np.einsum("ij,ij,j->ij", fg_output[:, i, :], cfinal[:, i, :], self.forgetGateWeights[-2, :])
            matrix += np.einsum("ij,ij,j->ij", ig_derivs[:, i, :], cpred_output[:, i, :], self.inputGateWeights[-2, :])
            matrix += np.einsum("ij,ij,j->ij", ig_output[:, i, :], cs_derivs[:, i, :], self.cellStateWeights[-2, :])
            matrix = np.einsum("ij,ij,ij->ij", og_output[:, i, :], ht_derivs[:, i, :], matrix)
            matrix += np.einsum("ij,ij,j->ij", og_derivs[:, i, :], activt[:, i, :], self.outputGateWeights[-2, :])
            delta_curr_layer[:, :, :] = np.einsum("ijk,ik->ijk", delta_curr_layer, matrix)

        fg_weight_gradients[:, :] = np.divide(fg_weight_gradients, inputs.shape[1])
        ig_weight_gradients[:, :] = np.divide(ig_weight_gradients, inputs.shape[1])
        og_weight_gradients[:, :] = np.divide(og_weight_gradients, inputs.shape[1])
        cs_weight_gradients[:, :] = np.divide(cs_weight_gradients, inputs.shape[1])
        return (fg_weight_gradients, ig_weight_gradients, og_weight_gradients, cs_weight_gradients), delta_curr_layer

    def inputShape(self):
        return None, None, self.nFeatures

    def outputShape(self, input_shape=None):
        return None, None, self.nUnits

    def applyWeightCorrections(self, weight_correction):
        self.forgetGateWeights[:, :] = np.add(self.forgetGateWeights, weight_correction[0])
        self.inputGateWeights[:, :] = np.add(self.inputGateWeights, weight_correction[1])
        self.outputGateWeights[:, :] = np.add(self.outputGateWeights, weight_correction[2])
        self.cellStateWeights[:, :] = np.add(self.cellStateWeights, weight_correction[3])

    def getWeights(self):
        return self.forgetGateWeights, self.inputGateWeights, self.outputGateWeights, self.cellStateWeights

    def setWeights(self, weights):
        self.forgetGateWeights[:, :] = weights[0]
        self.inputGateWeights[:, :] = weights[1]
        self.outputGateWeights[:, :] = weights[2]
        self.cellStateWeights[:, :] = weights[3]


class SimpleRNNLayer(Layer):
    """
    Layer comprising of simple RNN cells that feed the output as input to next step
    o_t-1 is the output from previous step, fed to the next step as input, as shown below
    output gate: o_t = sigmoid(Wo * X_t + Uo * o_t-1 + bo)
    """

    def __init__(self, num_units, num_features, output_gate_activation=Sigmoid(), init_wt=1E-2, init_state=0):
        """
        Initialize a simple RNN layer
        @param num_units:
        @param num_features:
        @param output_gate_activation:
        @param init_wt:
        @param init_state: Initial state fed to the RNN at t=0
        """
        self.nUnits = num_units
        self.nFeatures = num_features
        assert isinstance(output_gate_activation, Activation)
        self.outputGateActiv = output_gate_activation
        self.outputGateWeights = np.random.random((num_features + 2, num_units)) * init_wt
        self.initState = init_state


    def output(self, inputs, bias=None, **kwargs):
        """
        Output of the layer
        @param inputs: 3d ndarray of shape (#batches, #timesteps, #features)
        @param bias: None (for 0 bias) or a 3 d ndarray of shape (#batches, #timesteps, #units) to apply to each neuron
        @param kwargs: return_seq=True to return output for all timesteps,
        derivs=True to return derivatives,
        put_initial_state=True for initial state at t=0 to be added to output as first element. This means that
        output[:, i, :] will correspond to output for time step i-1
        @return: Tuple of all_outputs and derivatives. Depending upon kwargs, output will be a 3 dimensional ndarray
        of shape (#batches, #timesteps, #units) if return_sequences=True, or 2d ndarray of shape (#batches, #units) if
        return_sequences=False
        Derivative (if requested) will be 3d ndarray of shape (#batches, #timesteps, #units)
        """
        output_gate_inp = np.einsum("ijk,kl->ijl", inputs, self.outputGateWeights[0:-2, :])

        if bias:
            output_gate_inp = output_gate_inp + np.einsum("ijk,k->ijk", bias[:, :, :], self.outputGateWeights[-1, :])

        cell_state = np.ndarray((inputs.shape[0], self.nUnits), dtype=np.float)
        cell_state[:, :] = self.initState
        all_outputs, begin = None, None
        return_seq = kwargs.get("return_seq", False)
        put_initial_state = kwargs.get("put_initial_state", False)
        if return_seq:
            if put_initial_state:
                all_outputs = np.zeros((inputs.shape[0], inputs.shape[1]+1, self.nUnits), dtype=np.float)
                all_outputs[:, 0, :] = cell_state
                begin = 1
            else:
                all_outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)
                begin = 0
        derivs_required = kwargs.get("derivs", False)
        output_gate_derivs = None
        if derivs_required:
            output_gate_derivs = np.zeros((inputs.shape[0], inputs.shape[1], self.nUnits), dtype=np.float)

        for i in range(inputs.shape[0]):
            out_input = output_gate_inp[:, i, :] + np.einsum("ij,j->ij", cell_state, self.outputGateWeights[-2, :])
            cell_state = self.outputGateActiv.output(out_input)
            if return_seq:
                all_outputs[:, begin+i, :] = cell_state
            if derivs_required:
                output_gate_derivs[:, i, :] = self.outputGateActiv.deriv(out_input)

        if return_seq:
            return all_outputs, output_gate_derivs
        return cell_state, output_gate_derivs

    def deriv(self, inputs, bias=None):
        """
        Derivative of cell gates
        @param inputs: 3d ndarray of shape (#batches, #timesteps, #features)
        @param bias: None (for 0 bias) or a 3 d ndarray of shape (#batches, #timesteps, #units) to apply to each neuron
        @return: derivative of output date
        """
        return self.output(inputs, bias, derivs=True)[1:]

    def applyWeightsToInput(self, inputs, bias=None):
        pass

    def backPropagation(self, inputs, layer_deriv, delta_arr_layer, deriv_loss_wrt_outputs, bias=None):
        """
        Back propagation though time
        @param inputs: 3d ndarray of shape (#batches, #timesteps, #features)
        @param layer_deriv: (not used), will calculate gate derivatives
        @param delta_arr_layer: (will change its contents): dy_output_neuron/dy_layer_neuron.
        @param deriv_loss_wrt_outputs: dLoss/dy_output_neuron : 2D ndarray of shape (#batches, #output neurons)
        @param bias: None (for 0 bias) or a 3 d ndarray of shape (#batches, #timesteps, #units) to apply to each neuron
        @return: weight gradients and delta_arr_layer containing derivatives output network output w.r.t last layer's outputs
        """
        output, derivs = self.output(inputs, bias, derivs=True, return_seq=True, put_initial_state=True)
        weight_gradients = np.zeros(self.outputGateWeights.shape, dtype=np.float)
        delta_curr_layer = delta_arr_layer
        for i in range(inputs.shape[1]-1, -1, -1):
            # calculate weight gradients
            matrix = np.einsum("ij,ijk,ik->ik", deriv_loss_wrt_outputs,
                               delta_curr_layer,
                               derivs[:, i, :],
                               inputs[:, i, :])
            weight_gradients[0:-2, :] += np.einsum("ik,il->kl", matrix, inputs[:, i, :]).T
            # output[:, i, :] is the output of i-1 layer because put_initial_state=True
            weight_gradients[-2, :] += np.einsum("ik,ik->k", matrix, output[:, i, :])
            if bias:
                weight_gradients[-1, :] += np.einsum("ik,ik->k", matrix, bias[:, i, :])

            # calculate derivatives dy_l/dy_j for last layer, y_l : output, y_j: last layer's output
            delta_curr_layer[:, :, :] = np.einsum("ijk,ik,k->ijk", delta_curr_layer, derivs[:, i, :], self.outputGateWeights[-1, :])

        return np.divide(weight_gradients, inputs.shape[1]), delta_curr_layer

    def inputShape(self):
        return None, None, self.nFeatures

    def outputShape(self, input_shape=None):
        return None, None, self.nUnits

    def applyWeightCorrections(self, weight_correction):
        self.outputGateWeights = np.add(self.outputGateWeights, weight_correction)

    def getWeights(self):
        return self.outputGateWeights

    def setWeights(self, weights):
        self.outputGateWeights[:, :] = weights

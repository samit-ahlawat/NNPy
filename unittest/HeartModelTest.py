from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
import unittest
import os
import os.path
import logging
import src.lib.OptimizationAlgo as OA
import src.lib.NeuralNetwork as NN
import src.lib.Layer as Layer
import src.lib.Activation as AC
import src.lib.LossFunction as LF
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)


class HeartModelTest(unittest.TestCase):
    OPTIM_ALGO = OA.ADAM()
    ACTIVATION = AC.UnitActivation()

    def setUp(self):
        dataDir = "data"
        self.numEpochs = 15
        self.logger = logging.getLogger(self.__class__.__name__)
        train_data = pd.read_csv(os.path.join(dataDir, 'heart.csv'))
        train_data = self.transformData(train_data)
        train_rows = int(0.7 * train_data.shape[0])
        train_data_final = train_data.loc[0:train_rows, :]
        test_data = train_data.loc[train_rows + 1:, :].reset_index(drop=True)
        self.trainData = train_data_final
        self.testData = test_data

    def resultCol(self):
        return 'target'

    def transformData(self, data):
        data.loc[:, 'thal'] = data.loc[:, 'thal'].astype('category').cat.codes
        # normalize data
        data.loc[:, 'age'] = np.divide(data.loc[:, 'age'].values, 100.0)
        data.loc[:, 'trestbps'] = np.divide(data.loc[:, 'trestbps'].values, 200.0)
        data.loc[:, 'chol'] = np.divide(data.loc[:, 'chol'].values, 400.0)
        data.loc[:, 'thalach'] = np.divide(data.loc[:, 'thalach'].values, 200.0)
        data.loc[:, 'oldpeak'] = np.divide(data.loc[:, 'oldpeak'].values, 6.0)
        data.loc[:, 'slope'] = np.divide(data.loc[:, 'slope'].values, 3.0)
        data.loc[:, 'ca'] = np.divide(data.loc[:, 'ca'].values, 3.0)
        data.loc[:, 'thal'] = np.divide(data.loc[:, 'thal'].values, 3.0)
        data.loc[:, "restecg"] = np.divide(data.loc[:, "restecg"].values, 2.0)
        return data

    def neuralNetSetup(self, num_features):
        neuralnet = NN.SequentialNeuralNetwork(optim_algo=self.OPTIM_ALGO)
        input_layer = Layer.DenseLayer(num_inputs=num_features, num_neurons=6, network_output_neurons=1, activation=self.ACTIVATION)
        hidden_layer = Layer.DenseLayer(num_inputs=6, num_neurons=10, network_output_neurons=1, activation=self.ACTIVATION)
        output_layer = Layer.DenseLayer(num_inputs=10, num_neurons=1, network_output_neurons=1, activation=self.ACTIVATION)
        neuralnet.addLayer(input_layer)
        neuralnet.addLayer(hidden_layer)
        neuralnet.addLayer(output_layer)
        return neuralnet

    def trainNeuralNet(self, data):
        result = data.pop(self.resultCol()).values
        num_features = data.shape[1]
        self.nn = self.neuralNetSetup(num_features)
        xvals = data.values
        stats = self.nn.fit(xvals, result, epochs=self.numEpochs)
        data_dict = getattr(stats, "history", stats)
        data_dict["epoch"] = range(self.numEpochs)
        df = pd.DataFrame(data=data_dict)
        self.logger.info("Training stats: %s", df.to_string(index=False))

    def predict(self, data):
        result = data.pop(self.resultCol()).values
        xvals = data.values
        predict = self.nn.predict(xvals)
        data.loc[:, self.resultCol()] = result  # restore original value
        data.loc[:, 'raw_predict'] = predict
        data.loc[:, 'predict'] = data.loc[:, 'raw_predict'].apply(lambda x: 0 if x <= 0.5 else 1)
        data.loc[:, 'match'] = data.loc[:, self.resultCol()].eq(data.predict)
        return data

    def test_NNModel(self):
        self.trainNeuralNet(self.trainData)
        df = self.predict(self.testData)
        matched_rows = np.sum(df.loc[:, 'match'].values)
        perc_matched = float(matched_rows) / df.shape[0]
        self.logger.info('Matched rows: %d, total rows: %d, match percentage: %f', matched_rows, df.shape[0], 100*perc_matched)
        self.assertTrue(perc_matched > 0.5)


class HeartModelADAMUnitTest(HeartModelTest):
    OPTIM_ALGO = OA.ADAM()
    ACTIVATION = AC.UnitActivation()


class HeartModelRMSPropUnitTest(HeartModelTest):
    OPTIM_ALGO = OA.RMSProp(learning_rate=0.08)
    ACTIVATION = AC.UnitActivation()


class HeartModelSGDSigmoidTest(HeartModelTest):
    OPTIM_ALGO = OA.SimpleGradDescent(learning_rate=0.02)
    ACTIVATION = AC.Sigmoid()


class HeartModelADAMReluTest(HeartModelTest):
    # with ReLU need to use higher learning rates
    OPTIM_ALGO = OA.ADAM(learning_rate=0.1)
    ACTIVATION = AC.RectifiedLinear()


class HeartModelSGDReluTest(HeartModelTest):
    # with ReLU need to use higher learning rates
    OPTIM_ALGO = OA.SimpleGradDescent(learning_rate=0.4)
    ACTIVATION = AC.RectifiedLinear()


class HeartModelRMSPropReluTest(HeartModelTest):
    # with ReLU need to use higher learning rates
    OPTIM_ALGO = OA.RMSProp(learning_rate=0.1)
    ACTIVATION = AC.RectifiedLinear()


class TensorflowTest(HeartModelTest):
    def neuralNetSetup(self, num_features):
        neuralnet = tf.keras.Sequential()
        neuralnet.add(tf.keras.layers.Dense(6, input_shape=(num_features, ), activation="sigmoid"))
        neuralnet.add(tf.keras.layers.Dense(10, activation="sigmoid"))
        neuralnet.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        neuralnet.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())
        return neuralnet


class TensorflowTestSoftmax(HeartModelTest):
    def trainNeuralNet(self, data):
        result = data.pop(self.resultCol()).values
        res = np.zeros((result.shape[0], 2), dtype=np.bool)
        res[:, 0] = np.where(result == 0, 1, 0)
        res[:, 1] = np.where(result == 1, 1, 0)
        num_features = data.shape[1]
        self.nn = self.neuralNetSetup(num_features)
        xvals = data.values
        stats = self.nn.fit(xvals, res, epochs=self.numEpochs)
        data_dict = getattr(stats, "history", stats)
        data_dict["epoch"] = range(self.numEpochs)
        df = pd.DataFrame(data=data_dict)
        self.logger.info("Training stats: %s", df.to_string())

    def neuralNetSetup(self, num_features):
        neuralnet = tf.keras.Sequential()
        neuralnet.add(tf.keras.layers.Dense(6, input_shape=(num_features, ), activation="sigmoid"))
        neuralnet.add(tf.keras.layers.Dense(10, activation="sigmoid"))
        neuralnet.add(tf.keras.layers.Dense(2))
        neuralnet.add(tf.keras.layers.Softmax())
        neuralnet.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
        return neuralnet

    def predict(self, data):
        result = data.pop(self.resultCol()).values
        xvals = data.values
        predict = self.nn.predict(xvals)
        data.loc[:, self.resultCol()] = result  # restore original value
        data.loc[:, 'raw_predict_0'] = predict[:, 0]
        data.loc[:, 'raw_predict_1'] = predict[:, 1]
        data.loc[:, 'predict'] = data.loc[:, 'raw_predict_1'].apply(lambda x: 0 if x <= 0.5 else 1)
        data.loc[:, 'match'] = data.loc[:, self.resultCol()].eq(data.predict)
        return data


class NNPySoftmax(TensorflowTestSoftmax):
    OPTIM_ALGO = OA.SimpleGradDescent(learning_rate=0.05)
    ACTIVATION = AC.Sigmoid()

    def trainNeuralNet(self, data):
        result = data.pop(self.resultCol()).values
        num_features = data.shape[1]
        self.nn = self.neuralNetSetup(num_features)
        xvals = data.values
        stats = self.nn.fit(xvals, result, epochs=self.numEpochs)
        data_dict = getattr(stats, "history", stats)
        data_dict["epoch"] = range(self.numEpochs)
        df = pd.DataFrame(data=data_dict)
        self.logger.info("Training stats: %s", df.to_string())

    def neuralNetSetup(self, num_features):
        neuralnet = NN.SequentialNeuralNetwork(optim_algo=self.OPTIM_ALGO, loss_func=LF.SparseBinaryCrossEntropy(from_logits=False))
        input_layer = Layer.DenseLayer(num_inputs=num_features, num_neurons=6, network_output_neurons=2,
                                       activation=self.ACTIVATION)
        hidden_layer = Layer.DenseLayer(num_inputs=6, num_neurons=10, network_output_neurons=2,
                                        activation=self.ACTIVATION)
        output_layer_ua = Layer.DenseLayer(num_inputs=10, num_neurons=2, network_output_neurons=2,
                                           activation=AC.UnitActivation())
        softmax_layer = Layer.SoftmaxLayer(buckets=2)
        neuralnet.addLayer(input_layer)
        neuralnet.addLayer(hidden_layer)
        neuralnet.addLayer(output_layer_ua)
        neuralnet.addLayer(softmax_layer)
        return neuralnet


class NNPySoftmaxUnitActiv(NNPySoftmax):
    OPTIM_ALGO = OA.SimpleGradDescent(learning_rate=0.1)
    ACTIVATION = AC.UnitActivation()


if __name__ == "__main__":
    unittest.main()

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
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)


class TitanicModelTest(unittest.TestCase):
    OPTIM_ALGO = OA.ADAM()
    ACTIVATION = AC.UnitActivation()

    def setUp(self):
        dataDir = "data"
        self.numEpochs = 15
        self.logger = logging.getLogger(self.__class__.__name__)
        train_data = pd.read_csv(os.path.join(dataDir, 'titanic.csv'))
        train_data = self.transformData(train_data)
        train_rows = int(0.7 * train_data.shape[0])
        train_data_final = train_data.loc[0:train_rows, :]
        test_data = train_data.loc[train_rows + 1:, :].reset_index(drop=True)
        self.trainData = train_data_final
        self.testData = test_data

    def resultCol(self):
        return 'survived'

    def transformData(self, data):
        data.loc[:, 'sex'] = data.loc[:, 'sex'].astype('category').cat.codes
        data.loc[:, 'class'] = data.loc[:, 'class'].astype('category').cat.codes
        data.loc[data.loc[:, 'deck'].eq('unknown'), 'deck'] = pd.NaT
        data.loc[:, 'deck'] = data.loc[:, 'deck'].astype('category').cat.codes
        # convert NA values (which are -1 now) to 0
        data.loc[:, 'deck'] = np.add(data.loc[:, 'deck'].values, 1)
        data.loc[:, 'embark_town'] = data.loc[:, 'embark_town'].astype('category').cat.codes
        data.loc[:, 'alone'] = data.loc[:, 'alone'].astype('category').cat.codes
        # normalize data
        data.loc[:, 'age'] = np.divide(data.loc[:, 'age'].values, 100.0)
        data.loc[:, 'fare'] = np.divide(data.loc[:, 'fare'].values, 100.0)
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
        self.logger.info("Training stats: %s", df.to_string())

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


class TitanicModelADAMUnitTest(TitanicModelTest):
    OPTIM_ALGO = OA.ADAM()
    ACTIVATION = AC.UnitActivation()


class TitanicModelRMSPropUnitTest(TitanicModelTest):
    OPTIM_ALGO = OA.RMSProp(learning_rate=0.08)
    ACTIVATION = AC.UnitActivation()


class TitanicModelSGDSigmoidTest(TitanicModelTest):
    OPTIM_ALGO = OA.SimpleGradDescent(learning_rate=0.02)
    ACTIVATION = AC.Sigmoid()


class TitanicModelADAMReluTest(TitanicModelTest):
    # with ReLU need to use higher learning rates
    OPTIM_ALGO = OA.ADAM(learning_rate=0.1)
    ACTIVATION = AC.RectifiedLinear()


class TitanicModelSGDReluTest(TitanicModelTest):
    # with ReLU need to use higher learning rates
    OPTIM_ALGO = OA.SimpleGradDescent(learning_rate=0.1)
    ACTIVATION = AC.RectifiedLinear()


class TitanicModelRMSPropReluTest(TitanicModelTest):
    # with ReLU need to use higher learning rates
    OPTIM_ALGO = OA.RMSProp(learning_rate=0.1)
    ACTIVATION = AC.RectifiedLinear()


class TensorflowTest(TitanicModelTest):
    def neuralNetSetup(self, num_features):
        neuralnet = tf.keras.Sequential()
        neuralnet.add(tf.keras.layers.Dense(6, input_shape=(num_features, ), activation="sigmoid"))
        neuralnet.add(tf.keras.layers.Dense(10, activation="sigmoid"))
        neuralnet.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        neuralnet.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())
        return neuralnet


if __name__ == "__main__":
    unittest.main()

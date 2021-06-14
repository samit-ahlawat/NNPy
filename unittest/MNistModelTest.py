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
import src.lib.Metrics as Metrics
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)


class MNistModelTest(unittest.TestCase):
    def dataName(self):
        return "digits"

    def setUp(self):
        dataDir = "data"
        self.numEpochs = 10
        self.logger = logging.getLogger(self.__class__.__name__)
        name = self.dataName()
        data_file = os.path.join(dataDir, 'mnist', '%s_train_images.txt' % name)
        train_data_x = np.fromfile(data_file, dtype=np.uint8, sep=",").reshape((60000, 28, 28))
        data_file = os.path.join(dataDir, 'mnist', '%s_train_labels.txt' % name)
        train_data_y = np.fromfile(data_file, dtype=np.uint8, sep=",")
        data_file = os.path.join(dataDir, 'mnist', '%s_test_images.txt' % name)
        test_data_x = np.fromfile(data_file, dtype=np.uint8, sep=",").reshape((10000, 28, 28))
        data_file = os.path.join(dataDir, 'mnist', '%s_test_labels.txt' % name)
        test_data_y = np.fromfile(data_file, dtype=np.uint8, sep=",")
        data = (train_data_x, train_data_y, test_data_x, test_data_y)
        train_data_x, train_data_y, test_data_x, test_data_y = self.transformData(data)
        self.trainData = (train_data_x, train_data_y)
        self.testData = (test_data_x, test_data_y)

    def predict(self, data, nn):
        result = data[1]
        xvals = data[0]
        predict = nn.predict(xvals)
        df = pd.DataFrame(data={"sample": range(result.shape[0])})
        df.loc[:, "result"] = result  # restore original value
        df.loc[:, 'predict'] = np.argmax(predict, axis=1)
        df.loc[:, 'match'] = df.loc[:, "result"].eq(df.predict)
        return df

    def transformData(self, data):
        train_data_x = data[0]
        test_data_x = data[2]
        train_data_x = np.divide(train_data_x, 255.0)
        test_data_x = np.divide(test_data_x, 255.0)
        return train_data_x, data[1], test_data_x, data[3]

    def neuralNetSetup(self):
        flatten_layer = Layer.FlattenLayer()  # 28 x 28 -> 784 x 1
        ninput = 784
        noutput = 128
        dense_layer1 = Layer.DenseLayer(ninput, noutput, 10, activation=AC.RectifiedLinear(), init_wt=0.25)
        dropout_layer = Layer.DropoutLayer(dropout_fraction=0.2)
        dense_layer2 = Layer.DenseLayer(noutput, 10, 10)
        seq_network = NN.SequentialNeuralNetwork(loss_func=LF.SparseCategoricalCrossEntropy(from_logits=True),
                                                 optim_algo=OA.SimpleGradDescent(learning_rate=0.1),
                                                 metrics=[Metrics.Accuracy(probability=True, sparse=True)])
        seq_network.addLayer(flatten_layer)
        seq_network.addLayer(dense_layer1)
        seq_network.addLayer(dropout_layer)
        seq_network.addLayer(dense_layer2)
        return seq_network

    def tfNeuralNetSetup(self):
        flatten_layer = tf.keras.layers.Flatten(input_shape=(28, 28))  # 28 x 28 -> 784 x 1
        dense_layer1 = tf.keras.layers.Dense(128, activation="relu")
        dropout_layer = tf.keras.layers.Dropout(rate=0.2)
        dense_layer2 = tf.keras.layers.Dense(10)
        seq_network = tf.keras.models.Sequential([flatten_layer,
                                                  dense_layer1,
                                                  dropout_layer,
                                                  dense_layer2])
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        seq_network.compile(optimizer="adam", loss=loss_func, metrics=["accuracy"])
        return seq_network

    def trainNeuralNet(self, data, nn):
        stats = nn.fit(data[0], data[1], epochs=self.numEpochs)
        data_dict = getattr(stats, "history", stats)
        data_dict["epoch"] = range(self.numEpochs)
        df = pd.DataFrame(data=data_dict)
        self.logger.info("Training stats: %s", df.to_string(index=False))
        return nn

    def predictAndCompare(self, trained_nn):
        df = self.predict(self.testData, trained_nn)
        matched_rows = np.sum(df.loc[:, 'match'].values)
        perc_matched = float(matched_rows) / df.shape[0]
        self.logger.info('Matched rows: %d, total rows: %d, match percentage: %f', matched_rows, df.shape[0],
                         100 * perc_matched)
        self.assertTrue(perc_matched > 0.3)

    def test_NNModel(self):
        nn = self.neuralNetSetup()
        nn = self.trainNeuralNet(self.trainData, nn)
        return self.predictAndCompare(nn)

    def test_TFNNModel(self):
        nn = self.tfNeuralNetSetup()
        nn = self.trainNeuralNet(self.trainData, nn)
        return self.predictAndCompare(nn)


class FashionMNistModelTest(MNistModelTest):
    def dataName(self):
        return "fashion"

    def neuralNetSetup(self):
        flatten_layer = Layer.FlattenLayer()  # 28 x 28 -> 784 x 1
        ninput = 784
        noutput = 128
        dense_layer1 = Layer.DenseLayer(ninput, noutput, 10, init_wt=0.5)
        dense_layer2 = Layer.DenseLayer(noutput, 10, 10)
        seq_network = NN.SequentialNeuralNetwork(loss_func=LF.SparseCategoricalCrossEntropy(from_logits=True),
                                                 optim_algo=OA.ADAM(learning_rate=0.05),
                                                 metrics=[Metrics.Accuracy(probability=True, sparse=True)])
        seq_network.addLayer(flatten_layer)
        seq_network.addLayer(dense_layer1)
        seq_network.addLayer(dense_layer2)
        return seq_network

    def tfNeuralNetSetup(self):
        flatten_layer = tf.keras.layers.Flatten(input_shape=(28, 28))  # 28 x 28 -> 784 x 1
        dense_layer1 = tf.keras.layers.Dense(128, activation="relu")
        dense_layer2 = tf.keras.layers.Dense(10)
        seq_network = tf.keras.models.Sequential([flatten_layer,
                                                  dense_layer1,
                                                  dense_layer2])
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        seq_network.compile(optimizer="adam", loss=loss_func, metrics=["accuracy"])
        return seq_network



if __name__ == "__main__":
    unittest.main()

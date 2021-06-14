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
from tensorflow.keras import models, layers, losses

logging.basicConfig(level=logging.DEBUG)


class CNNModelTest(unittest.TestCase):
    THRESHOLD = 0.4
    INIT_WT = 0.25

    def dataName(self):
        return "digits"

    def setUp(self):
        dataDir = "data"
        self.numEpochs = 25
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
        keep = train_data_x.shape[0]//20
        train_data_x = train_data_x[0:keep, :, :]
        test_data_x = np.divide(test_data_x, 255.0)
        return train_data_x[..., np.newaxis], data[1][0:keep], test_data_x[..., np.newaxis], data[3]

    def neuralNetSetup(self):
        # (28, 28, 1) -> (26, 26, 10)
        conv_layer1 = Layer.ConvolutionLayer((3, 3), 1, 10, init_wt=self.INIT_WT)
        # (26, 26, 10) -> (13, 13, 10)
        pooling_layer1 = Layer.PoolingLayer2D((2, 2))  # max pooling by default
        # (13, 13, 10) -> (10, 10, 20)
        conv_layer2 = Layer.ConvolutionLayer((4, 4), 10, 20, init_wt=self.INIT_WT)
        # (10, 10, 20) -> (5, 5, 20)
        pooling_layer2 = Layer.PoolingLayer2D((2, 2))  # max pooling by default
        conv_layer3 = Layer.ConvolutionLayer((3, 3), 20, 20, init_wt=self.INIT_WT)
        # (3, 3, 20) -> (180)
        flatten_layer = Layer.FlattenLayer()
        # 180
        dense_layer1 = Layer.DenseLayer(180, 20, 10, activation=AC.RectifiedLinear(), init_wt=self.INIT_WT)
        dense_layer2 = Layer.DenseLayer(20, 10, 10, init_wt=self.INIT_WT)
        seq_network = NN.SequentialNeuralNetwork(loss_func=LF.SparseCategoricalCrossEntropy(from_logits=True),
                                                 optim_algo=OA.ADAM(learning_rate=0.05),
                                                 metrics=[Metrics.Accuracy(probability=True, sparse=True)])
        seq_network.addLayer(conv_layer1)
        seq_network.addLayer(pooling_layer1)
        seq_network.addLayer(conv_layer2)
        seq_network.addLayer(pooling_layer2)
        seq_network.addLayer(conv_layer3)
        seq_network.addLayer(flatten_layer)
        seq_network.addLayer(dense_layer1)
        seq_network.addLayer(dense_layer2)
        return seq_network

    def tfNeuralNetSetup(self):
        model = models.Sequential()
        model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # 26 X 26 X 10
        model.add(layers.MaxPooling2D((2, 2)))  # 13 X 13 X 10
        model.add(layers.Conv2D(20, (4, 4), activation='relu'))  # 10 X 10 X 20
        model.add(layers.MaxPooling2D((2, 2)))  # 5 X 5 X 20
        model.add(layers.Conv2D(20, (3, 3), activation='relu'))  # 3 X 3 X 20
        model.add(layers.Flatten())
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(10))
        model.summary()
        model.compile(optimizer='adam',
                      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

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
        self.assertTrue(perc_matched > self.THRESHOLD)

    def test_NNModel(self):
        nn = self.neuralNetSetup()
        nn = self.trainNeuralNet(self.trainData, nn)
        return self.predictAndCompare(nn)

    def test_TFNNModel(self):
        nn = self.tfNeuralNetSetup()
        nn = self.trainNeuralNet(self.trainData, nn)
        return self.predictAndCompare(nn)


class FashionMNistCNNModelTest(CNNModelTest):
    THRESHOLD = 0.25
    INIT_WT = 0.75

    def dataName(self):
        return "fashion"

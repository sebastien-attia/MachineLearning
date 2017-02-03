import sys
sys.path.append('../scripts')

import numpy as np
import unittest
from neural_network import NeuralNetwork

class TestStringMethods(unittest.TestCase):
    def test_NN_1(self):
        weights = [
            np.array([
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.]
            ]),
            np.array([
                [0.30, 0.29, 0.28, 0.27, 0.26],
                [0.25, 0.24, 0.23, 0.22, 0.21],
                [0.20, 0.19, 0.18, 0.17, 0.16],
                [0.15, 0.14, 0.13, 0.12, 0.11],
                [0.10, 0.09, 0.08, 0.07, 0.06],
                [0.05, 0.04, 0.03, 0.02, 0.01]
            ]),
            np.array([
                [75., 65., 55., 45., 35., 25., 15.]
            ])
        ]

        nn = NeuralNetwork([2, 4, 6, 1], thetas=weights)

        self.assertEqual((4,3), nn.thetas[0].shape)
        self.assertEqual((6,5), nn.thetas[1].shape)
        self.assertEqual((1,7), nn.thetas[2].shape)

        training_set = (
            (np.array([0, 0]), np.array([0])),
            (np.array([0, 1]), np.array([1])),
            (np.array([1, 0]), np.array([1])),
            (np.array([1, 1]), np.array([0])),
        )

        #print(nn.thetas)
        nn.train(training_set, 0.1, debug=True)
        #print(nn.thetas)

        print('PREDICT: ', nn.predict(np.array([1, 1])) )
        #print('PREDICT: ', nn.predict(np.array([0, 1])) )
        #print('PREDICT: ', nn.predict(np.array([1, 0])) )
        #print('PREDICT: ', nn.predict(np.array([0, 0])) )

        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()

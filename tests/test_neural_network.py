import sys
sys.path.append('../scripts')

import numpy as np
import unittest
from neural_network import NeuralNetwork

class TestStringMethods(unittest.TestCase):
    def test_NN_1(self):
        nn = NeuralNetwork([2, 10, 5, 1], init_epsilon=1)

        self.assertEqual((10,3), nn.thetas[0].shape)
        self.assertEqual((5,11), nn.thetas[1].shape)
        self.assertEqual((1,6), nn.thetas[2].shape)

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
        print('PREDICT: ', nn.predict(np.array([0, 1])) )
        print('PREDICT: ', nn.predict(np.array([1, 0])) )
        print('PREDICT: ', nn.predict(np.array([0, 0])) )

        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()

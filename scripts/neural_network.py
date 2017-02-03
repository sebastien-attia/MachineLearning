import numpy as np

from functions_library import (
    sigmoid, derivative_sigmoid,
    least_square_error, least_square_error_gradient
)

class NeuralNetwork:
    def __init__(self, layers_def, activation_fct=(sigmoid, derivative_sigmoid), cost_fct=(least_square_error, least_square_error_gradient), thetas=None, init_epsilon=1):
        '''
        layers_def is an array defining the number of neurons in each layer.
        layers_def[0] is the input layer.
        layers_def[len(layers_def)-1] is the output layer.
        The layers between are the hidden layers.

        The activation
        '''

        def compute_rand(i, j, init_epsilon):
            return np.random.rand(i, j)*2*init_epsilon-init_epsilon

        self.layers_def = layers_def

        if thetas:
            self.thetas = thetas
            self._check_thetas_shape(layers_def, thetas)
        else:
            self.thetas = [compute_rand(layers_def[l], layers_def[l-1]+1, init_epsilon)  for l in range(1, len(layers_def))]

        self.activation_fct = activation_fct[0]
        self.derivative_activation_fct = activation_fct[1]
        self.cost_fct = cost_fct[0]
        self.gradient_cost_fct = cost_fct[1]

    def train(self, data_set, learning_rate, mini_batch_size=1, reg_lambda=0, debug=False):
        mini_batches = [data_set[x:x+mini_batch_size] for x in range(0, len(data_set), mini_batch_size)]

        for mini_batch in mini_batches:
            self._train_mini_batch(mini_batch, learning_rate, reg_lambda, debug)

        return self._cost(data_set, reg_lambda)

    def predict(self, input):
        result = self._forward_propagation(input)
        return self._get_output(result)

    def _train_mini_batch(self, mini_batch, learning_rate, reg_lambda, debug=False):
        Deltas = list(range(len(self.thetas)))
        for x, y in mini_batch:
            output_layers = self._forward_propagation(x)
            size_mini_batch = len(mini_batch)
            self._backpropagation(x, y, output_layers, Deltas, reg_lambda, size_mini_batch, debug)

        for (i, theta) in enumerate(self.thetas):
            self.thetas[i] = self.thetas[i] - learning_rate * Deltas[i]

    def _forward_propagation(self, x, thetas=None, layer=None):
        if not thetas:
            thetas = self.thetas

        z = x
        output_layers = list()
        for (i, theta) in enumerate(thetas):
            a = np.append([1], z)
            z = np.dot(theta, a)
            a = self.activation_fct( z )
            output_layers.append((z, a))
            z = a

            if layer and i == layer:
                break

        return output_layers

    def _backpropagation(self, x, y, output_layers, Deltas, reg_lambda, size_mini_batch, debug):
        delta = None
        is_first = True

        for (layer, theta) in reversed(list(enumerate(self.thetas))):
            if layer == 0:
                break

            input_current_layer = output_layers[layer][0]

            if is_first:
                output_current_layer = output_layers[layer][1]
                delta = self.gradient_cost_fct(y, output_current_layer)
                is_first = False
            else:
                delta = np.dot(np.transpose(self._remove_bias_column(self.thetas[layer+1])), delta) * self.derivative_activation_fct(input_current_layer)

            output_previous_layer = np.append([1], output_layers[layer-1][1])
            main_term = np.outer(delta, np.transpose(output_previous_layer))
            reg_term = reg_lambda * self._zero_bias_thetas(theta)

            if debug:
                self._gradient_checking(x, y, reg_lambda, layer, main_term, reg_term)

            Deltas[layer] = Deltas[layer] + (main_term + reg_term)/size_mini_batch

    def _cost(self, data_set, reg_lambda, thetas=None):
        if not thetas:
            thetas = self.thetas

        m = len(data_set)
        def output(x):
            return self._get_output(self._forward_propagation(x, thetas))

        main_cost = -sum([self.cost_fct(y, output(x)) for x, y in data_set])/m

        def _sum_theta_square(theta):
            shape = theta.shape
            result = 0
            for i in range(shape[0]):
                for j in range(1, shape[1]):
                    result += theta[i][j]
            return result

        reg = reg_lambda * sum([_sum_theta_square(t) for t in thetas])/(2*m)
        result = main_cost + reg
        return np.array([result, main_cost, reg])

    def _zero_bias_thetas(self, theta):
        result = np.copy(theta)
        result[:, 0] = 0
        return result

    def _remove_bias_column(self, theta):
        return np.delete(np.copy(theta), 0, 1)

    def _gradient_checking(self, x, y, reg_lambda, layer, main_term, reg_term):
        epsilon = 0.01
        precision = np.array([ 0.001, 0.001, 0.001 ])

        lthetas = list(self.thetas)

        data_set = ((x, y),)

        shape = lthetas[layer].shape

        for i in range(shape[0]):
            for j in range(shape[1]):
                lthetas[layer] = np.copy(self.thetas[layer])
                lthetas[layer][i][j] += epsilon
                h_p_e = self._cost(data_set, reg_lambda, lthetas)

                lthetas[layer] = np.copy(self.thetas[layer])
                lthetas[layer][i][j] -= epsilon
                h_m_e = self._cost(data_set, reg_lambda, lthetas)

                exp_partial_derivative = (h_p_e - h_m_e)/(2.*epsilon)

                main_pd = main_term[i, j]
                reg_pd = reg_term[i, j]
                partial_derivative = np.array([ main_pd+reg_pd, main_pd, reg_pd ])

                err = np.less(
                    np.fabs(partial_derivative -  exp_partial_derivative),
                    precision
                )

    def _get_output(self, output_layers):
        return output_layers[len(output_layers)-1][1]

    def _add_bias_node(self, array):
        return np.append([1], array)

    def _check_thetas_shape(self, layers_def, thetas):
        if len(thetas) != (len(layers_def)-1):
            raise RuntimeError("Not valid dimension for the weights: expected[%s], get[%s]" % ((len(layers_def)-1), len(thetas)))

        for l in range(1, len(layers_def)):
            expected_shape = (layers_def[l], layers_def[l-1]+1)
            if thetas[l-1].shape != expected_shape:
                raise RuntimeError("For the layer [%s], the expected dim is [%s], but get [%s]" % (l, expected_shape, thetas[l].shape))

import numpy as np

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def derivative_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def least_square_error(y, output):
    diff = output - y
    return np.dot(np.transpose(diff), diff)/2

def least_square_error_gradient(y, output):
    return (output - y)

class NeuralNetwork:
    def __init__(self, layers_def, activation_fct=(sigmoid, derivative_sigmoid), cost_fct=(least_square_error, least_square_error_gradient), init_epsilon=1):
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

        self.thetas = [compute_rand(layers_def[l], layers_def[l-1]+1, init_epsilon)  for l in range(1, len(layers_def))]
        self.activation_fct = activation_fct[0]
        self.derivative_activation_fct = activation_fct[1]
        self.cost_fct = cost_fct[0]
        self.gradient_cost_fct = cost_fct[1]

    def train(self, data_set, learning_rate, mini_batch_size=1, reg_lambda=0, debug=False):
        mini_batches = [data_set[x:x+mini_batch_size] for x in range(0, len(data_set), mini_batch_size)]

        for mini_batch in mini_batches:
            self._train_mini_batch(mini_batch, learning_rate, reg_lambda, debug)

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

        for (i, theta) in reversed(list(enumerate(self.thetas))):
            if i == 0:
                break

            input_current_layer = output_layers[i][0]

            if is_first:
                output_current_layer = output_layers[i][1]
                delta = self.gradient_cost_fct(y, output_current_layer)
                is_first = False
            else:
                delta = np.dot(np.transpose(self._remove_bias_column(self.thetas[i+1])), delta) * self.derivative_activation_fct(input_current_layer)

            if debug:
                self._gradient_checking(x, i, delta)

            output_previous_layer = np.append([1], output_layers[i-1][1])
            term_1 = np.outer(delta, np.transpose(output_previous_layer))
            term_2 = reg_lambda * self._zero_bias_thetas(theta)

            Deltas[i] = Deltas[i] + (term_1 + term_2)/size_mini_batch

    def _zero_bias_thetas(self, theta):
        result = np.copy(theta)
        result[:, 0] = 0
        return result

    def _remove_bias_column(self, theta):
        return np.delete(np.copy(theta), 0, 1)

    def _gradient_checking(self, x, layer, delta):
        lthetas = self.thetas[layer]
        #print(len(lthetas))
        #print(lthetas)

    def _get_output(self, output_layers):
        return output_layers[len(output_layers)-1][1]

    def _add_bias_node(self, array):
        return np.append([1], array)

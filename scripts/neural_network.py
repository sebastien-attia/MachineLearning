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
    def __init__(self, layers_def, activation_fct=(sigmoid, derivative_sigmoid), cost_fct=(least_square_error, least_square_error_gradient)):
        '''
        layers_def is an array defining the number of neurons in each layer.
        layers_def[0] is the input layer.
        layers_def[len(layers_def)-1] is the output layer.
        The layers between are the hidden layers.

        The activation
        '''
        self.layers_def = layers_def

        self.thetas = [np.random.rand(layers_def[l], layers_def[l-1]+1) for l in range(1, len(layers_def))]
        self.activation_fct = activation_fct[0]
        self.derivative_activation_fct = activation_fct[1]
        self.cost_fct = cost_fct[0]
        self.gradient_cost_fct = cost_fct[1]

    def train(data_set, learning_rate, mini_batch_size=1, reg_lambda=0, debug=False):
        mini_batches = np.array_split(data_set, mini_batch_size)
        for mini_batch in mini_batches:
            _train_mini_batch(mini_batch, learning_rate, reg_lambda, debug)

    def _train_mini_batch(mini_batch, learning_rate, reg_lambda, debug=False):
        Deltas = list(range(len(self.thetas)))
        for x, y in mini_batch:
            output_layers = self._compute_forward_propagation(x)
            size_mini_batch = len(mini_batch)
            self._compute_backpropagation(y, output_layers, Deltas, reg_lambda, size_mini_batch)

        for theta, Delta in zip(self.thetas, Deltas):
            theta = theta - alpha * Delta

    def _compute_forward_propagation(self, x, layer=None):
        a = x
        output_layers = list()
        for theta in self.thetas:
            a = np.append([1], a)
            z = np.dot(theta, a)
            a = self.activation_fct(z)
            output_layers.append((z, a))

        return output_layers

    def _compute_backpropagation(self, y, output_layers, Deltas, reg_lambda, size_mini_batch):
        delta = None
        is_first = True

        for (i, theta) in reversed(list(enumerate(self.thetas))):
            if i == 1:
                break

            input_current_layer = output_layers[i][0]
            output_current_layer = output_layers[i][1]

            output_previous_layer = output_layers[i-1][1]

            if is_first:
                delta = self.gradient_cost_fct(y, input_current_layer)
                is_first = false
            else:
                output_previous_layer = output_layers[i-1]
                delta = np.dot(np.transpose(theta), delta) * self.derivative_activation_fct(output_current_layer)

            Deltas[i] = Deltas[i] + (delta * np.transpose(output_previous_layer) + reg_lambda * self._modified_theta(theta))/size_mini_batch

    def _modified_theta(self, theta):
        return theta[:, 0] = 0

import numpy as np


class NN:
    def __init__(self, n_layers, layer_size, input_size, output_size):
        self.layers = []
        for i in n_layers:
            self.layers.append(Layer())

class Activation:
    def __init__(self):

class Layer:
    def __init__(self, in_dim, out_dim, activation_fun):
        self.weights = np.full((in_dim, out_dim), 1 / (in_dim * out_dim))
        self.n_inputs = in_dim
        self.n_outputs = out_dim
        self.biases = np.full(out_dim, 1 / (in_dim * out_dim))
        self.activation = activation_fun
        self.current = None
        self.out = None

    def forward(self, input_data):
        self.current = input_data
        self.out = self.activation(self.current.dot(self.weights) + self.biases)

    def backward(self):
import numpy as np

def ReLUBackward(y, err):
    return (y > 0) * err
def ReLU(x):
    return np.maximum(0, x)
def SoftMaxBack(y, err):
    return err.dot(np.diagflat(y) - y.T.dot(y))
def SoftMax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1)
class NN:
    def __init__(self, n_layers, layer_size, input_size, output_size, classification=False, n_epochs=10, batch_size=64):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        if classification:
            self.activation_back = SoftMaxBack
            self.activation = SoftMax
        else:
            self.activation_back = ReLUBackward
            self.activation = ReLU
        self.layers = []
        for i in n_layers:
            self.layers.append(Layer())
    def forward(self, input_data):
        temp = input_data.copy()
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp

    def backward(self, err):
        temp = err
        for layer in self.layers:
            temp = layer.backward(temp)

    def train(self, X, y, something, idk):
        for epoch in range(self.n_epochs):
            for i in range(len(y)//self.batch_size):
                to = (i+1) * self.batch_size if (i+1) * self.batch_size < len(y) else len(y) - 1
                X_train = X[i * self.batch_size:to, :]
                prediction = self.forward(X_train)
                self.backward(prediction - y[i * self.batch_size:to, :])

class Layer:
    def __init__(self, in_dim, out_dim, activation_fun, activation_back, lr):
        self.weights = np.full((in_dim, out_dim), 1 / (in_dim * out_dim))
        self.n_inputs = in_dim
        self.n_outputs = out_dim
        self.biases = np.full(out_dim, 1 / (in_dim * out_dim))
        self.activation = activation_fun
        self.activation_back = activation_back
        self.lr = lr
        self.current = None
        self.out = None

    def forward(self, input_data):
        self.current = input_data
        self.out = self.activation(self.current.dot(self.weights) + self.biases)

    def backward(self, err):
        activation_err = self.activation_back(self.out, err)
        gradient = self.current.T.dot(activation_err)
        p_err = self.weights.dot(activation_err.T).T
        self.weights += self.lr * gradient
        self.biases += self.lr * activation_err
        return p_err




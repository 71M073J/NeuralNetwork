import numpy as np


def Sigmoid(x):
    return 1/(1 + np.exp(-x))

def SigmoidBack(y, err):
    return err * (y * (1 - y))

def ReLUBackward(y, err):
    return err * (y > 0)


def ReLU(x):
    return np.maximum(0, x)


def SoftMaxBack(y, err):
    #SM = y.reshape((-1, 1))
    #jac = np.diagflat(y) - np.dot(SM, SM.T)
    #err.dot((np.eye(y.shape[0]) - y.T.dot(y)))
    return (np.diagflat(y) - y.T.dot(y)).dot(err)


def SoftMax(x):
    x_exp = np.exp(x - np.max(x, axis=0))
    r = x_exp.T / np.sum(x_exp, axis=0)
    r[r == np.nan] = 0
    return r

class NN:
    def __init__(self, n_hidden_layers, layer_size, input_size, output_size, classification=False, n_epochs=10,
                 batch_size=64, lr=0.01):
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        if classification:
            ...
            # self.activation_back = SoftMaxBack
            # self.activation = SoftMax
        else:
            self.activation_back = ReLUBackward
            self.activation = ReLU

        self.layers = []
        self.layers.append(Layer(input_size, layer_size, self.activation, self.activation_back, self.lr))
        for i in range(n_hidden_layers - 1):
            self.layers.append(Layer(layer_size, layer_size, self.activation, self.activation_back, self.lr))
        #self.layers.append(Layer(layer_size, output_size, self.activation, self.activation_back, self.lr))
        self.layers.append(Layer(layer_size, output_size, SoftMax, SoftMaxBack, self.lr))
        #self.layers.append(Layer(layer_size, output_size, Sigmoid, SigmoidBack, self.lr))

    def forward(self, input_data):
        temp = input_data.copy()
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp

    def backward(self, err):
        temp = err
        for layer in reversed(self.layers):
            temp = layer.backward(temp)

    def train(self, X, y, something, idk):
        for epoch in range(self.n_epochs):
            if epoch % 20 == 0: print(f"epoch {epoch}")
            for i in range((len(y) // self.batch_size) + 1):
                if i * self.batch_size == len(y): break
                to = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(y) else len(y)
                X_train = X[i * self.batch_size:to, :]
                prediction = self.forward(X_train)
                self.backward((y[i * self.batch_size:to] - prediction.T).T)


class Layer:
    def __init__(self, in_dim, out_dim, activation_fun, activation_back, lr):
        self.weights = np.random.random((in_dim, out_dim))/(in_dim + out_dim)
        self.n_inputs = in_dim
        self.n_outputs = out_dim
        self.biases = np.random.random((1, out_dim))/(in_dim + out_dim)
        self.activation = activation_fun
        self.activation_back = activation_back
        self.lr = lr
        self.current = None
        self.out = None

    def forward(self, input_data):
        self.current = input_data
        self.out = self.activation(self.current.dot(self.weights) + self.biases)
        return self.out

    def backward(self, err):
        activation_err = self.activation_back(self.out, err)
        gradient = self.current.T.dot(activation_err)
        p_err = self.weights.dot(activation_err.T).T
        self.weights += self.lr * gradient
        self.biases += self.lr * activation_err.mean(axis=0)
        return p_err


if __name__ == "__main__":
    data = np.random.randint(1, 4, (1024, 2))
    y = data[:, 0] > 2
    #data = np.array([[1,2],[2,3],[3,4],[4,5]])
    #y = np.array((0,0,1,1))
    # input = np.array([[1, 2]])
    # target = np.array([[0, 1, 0]])
    # norm = np.linalg.norm(input, axis=1)
    # input_n = input / norm
    # data = input_n
    # y = target

    net = NN(4, 5, 2, 1, n_epochs=200, lr=0.01)
    net.train(data, y, None, None)
    print(net.forward(np.array([[3, 4],[0,0], [-4,4], [4,1]])))
    print(net.forward(np.array([[4, 3], [0,1]])))
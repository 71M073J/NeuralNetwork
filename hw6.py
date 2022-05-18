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
    return err.dot(np.diagflat(y) - y.T.dot(y))


def SoftMax(x):
    x_exp = np.exp(x - np.max(x))
    r = x_exp / np.sum(x_exp, axis=0)
    #r[r == np.nan] = 0
    return r

class NN:
    def __init__(self, n_hidden_layers, layer_size, input_size, output_size, classification=False,
                 batch_size=64, lr=0.01):
        self.lr = lr
        self.batch_size = batch_size
        if classification:
            ...
            # self.activation_back = SoftMaxBack
            # self.activation = SoftMax
        else:
            self.activation_back = ReLUBackward
            self.activation = ReLU

        self.layers = []
        if n_hidden_layers > 0:
            self.layers.append(Layer(input_size, layer_size, self.activation, self.activation_back, self.lr, self.batch_size))
            for i in range(n_hidden_layers - 1):
                self.layers.append(Layer(layer_size, layer_size, self.activation, self.activation_back, self.lr, self.batch_size))
            #self.layers.append(Layer(layer_size, output_size, self.activation, self.activation_back, self.lr))
            #self.layers.append(Layer(layer_size, output_size, SoftMax, SoftMaxBack, self.lr, self.batch_size))
            self.layers.append(Layer(layer_size, output_size, Sigmoid, SigmoidBack, self.lr, self.batch_size))
        else:
            self.layers.append(Layer(input_size, output_size, Sigmoid, SigmoidBack, self.lr))
    def forward(self, input_data):
        temp = input_data.copy()
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp

    def backward(self, err):
        temp = err
        for layer in reversed(self.layers):
            temp = layer.backward(temp)

    def train(self, X, y, n_epochs, test=None):
        for epoch in range(n_epochs):
            if epoch % 100 == 0: print(f"epoch {epoch}")
            for i in range((len(y) // self.batch_size) + 1):
                if i * self.batch_size == len(y): break
                to = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(y) else len(y)
                X_train = X[i * self.batch_size:to, :]
                prediction = self.forward(X_train)
                self.backward((y[i * self.batch_size:to] - prediction.T).T)


class Layer:
    def __init__(self, in_dim, out_dim, activation_fun, activation_back, lr, batch):
        self.weights = np.random.random((in_dim, out_dim))/(in_dim + out_dim)
        self.n_inputs = in_dim
        self.batch_size = batch
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
        pred_err = np.zeros((err.shape[0], self.n_inputs))
        for i in range(err.shape[0]):
            activation_err = self.activation_back(self.out[i], err[i])
            gradient = self.current[i, :].reshape(-1, 1).dot(np.atleast_2d(activation_err))
            pred_err[i, :] = self.weights.dot(activation_err.T).T
            self.weights += self.lr * gradient / self.batch_size
            self.biases += self.lr * activation_err / self.batch_size
        #print(self.biases)
        return pred_err


if __name__ == "__main__":
    data = np.random.randint(-5, 5, (5024, 2))
    y = (data[:, 0] + data[:,1]) > 1
    #data = np.array([[1,2],[2,3],[3,4],[4,5]])
    #y = np.array((0,0,1,1))
    # input = np.array([[1, 2]])
    # target = np.array([[0, 1, 0]])
    # norm = np.linalg.norm(input, axis=1)
    # input_n = input / norm
    # data = input_n
    # y = target

    #net = NN(1, 25, 2, 1, lr=0.01)
    #net.train(data, y, 200)
    #print(net.forward(np.array([[3, 0],[0,1], [-4,4], [4,1]])))
    #print(net.forward(np.array([[2, 0],[1,0],[0,2], [0,4], [0,3],[4,0],[3,0]])))

    #quit()
    net = NN(3,25,2, 1, lr=0.7)
    data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
    y = np.array((0,1,1,0))
    net.train(data, y, n_epochs=2000)
    print(net.forward(data))

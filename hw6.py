import numpy as np


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SigmoidBack(y, err):
    # return err * (y * (1 - y))
    sig = Sigmoid(y)
    return err * (sig * (1 - sig))


def cross_E(y_true, y_pred):  # CE
    return -np.sum(y_true * np.log(y_pred + 10 ** -100))


def cross_E_grad(y_true, y_pred):  # CE derivative
    return -y_true / (y_pred + 10 ** -100)


def ReLUBackward(y, err):
    return err * (y > 0)


def ReLU(x):
    return np.maximum(0, x)


def SoftMaxBack(y, err):
    # I = np.eye(y.shape[0])
    activation_err = np.zeros(err.shape)
    for i in range(err.shape[0]):
        activation_err[i] = err.dot(np.diagflat(y[i]) - y[i].T.dot(y[i]))
        ...
    # return err.dot(SoftMax(y) * (I - SoftMax(y).T)
    # SM = y.reshape((-1, 1))
    # jac = np.diagflat(y) - np.dot(SM, SM.T)
    # err.dot((np.eye(y.shape[0]) - y.T.dot(y)))
    return err.dot(np.diagflat(y) - y.T.dot(y))


def SoftMax(x):
    x_exp = np.exp(x - np.max(x, axis=0))
    r = x_exp.T / np.sum(x_exp, axis=0)
    # r[r == np.nan] = 0
    return r


def Linear(x):
    return x


def LinearBack(y, err):
    return err


class NN:
    def __init__(self, n_hidden_layers, layer_sizes, input_size, output_size, classification=False,
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
            self.layers.append(
                Layer(input_size, layer_sizes[0], self.activation, self.activation_back, self.lr, self.batch_size))
            for i in range(1, len(layer_sizes)):
                self.layers.append(
                    Layer(layer_sizes[i - 1], layer_sizes[i], self.activation, self.activation_back, self.lr, self.batch_size))
                # self.layers.append(Layer(layer_size, output_size, self.activation, self.activation_back, self.lr))
                # self.layers.append(Layer(layer_size, output_size, SoftMax, SoftMaxBack, self.lr, self.batch_size))
            self.layers.append(Layer(layer_sizes[-1], output_size, Linear, LinearBack, self.lr, self.batch_size))
        else:
            self.layers.append(Layer(input_size, output_size, Linear, LinearBack, self.lr, self.batch_size))

    def forward(self, input_data):
        temp = input_data.copy()
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp

    def backward(self, err):
        temp = err
        for layer in reversed(self.layers):
            temp = layer.backward(temp)

    def train(self, X, y, n_epochs, test=None, verbose=False):
        for epoch in range(n_epochs):
            if verbose and epoch % 100 == 0: print(f"epoch {epoch}")
            # else: print(epoch, end="")
            for i in range((len(y) // self.batch_size) + 1):
                if i * self.batch_size == len(y): break
                to = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(y) else len(y)
                X_train = X[i * self.batch_size:to, :]
                prediction = self.forward(X_train)
                # za regresijo
                self.backward((y[i * self.batch_size:to] - prediction.T).T)  # odvod MSE
                # self.backward(delta_cross_entropy(prediction, y[i * self.batch_size:to]))


class Layer:
    def __init__(self, in_dim, out_dim, activation_fun, activation_back, lr, batch):
        np.random.seed(1)
        self.weights = np.random.random((in_dim, out_dim)) / (in_dim * out_dim)
        self.n_inputs = in_dim
        self.batch_size = batch
        self.n_outputs = out_dim
        self.biases = np.random.random((1, out_dim)) / (in_dim * out_dim)
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
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.biases.shape)
        online_update = True
        for i in range(err.shape[0]):
            activation_err = self.activation_back(self.out[i], err[i])
            gradient = self.current[i, :].reshape(-1, 1).dot(np.atleast_2d(activation_err))
            pred_err[i, :] = self.weights.dot(activation_err.T).T
            if online_update:
                self.weights += self.lr * gradient / self.batch_size
                self.biases += self.lr * activation_err / self.batch_size
            else:
                dw += self.lr * gradient / self.batch_size
                db += self.lr * activation_err / self.batch_size
        if not online_update:
            self.weights += dw
            self.biases += db



        # print(temp, self.weights)
        # this gives the same results...?
        # just change softmax back to go case by case
        # activation_err = self.activation_back(self.out, err)
        # gradient = self.current.T.dot(np.atleast_2d(activation_err))
        # pred_err = self.weights.dot(activation_err.T).T
        # self.weights += self.lr * gradient/self.batch_size
        # self.biases += self.lr * activation_err.sum(axis=0)/self.batch_size

        # print(self.biases)
        return pred_err


class ANNRegression:
    def __init__(self, units, lambda_, lr=5, n_epochs=5000):
        self.layers = units
        self.lambda_ = lambda_
        self.model = None
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        self.model = NN(len(self.layers), self.layers, X.shape[1], 1, lr=self.lr)
        self.model.train(X, y, self.n_epochs)
        return self

    def predict(self, X):
        return self.model.forward(X).flatten()

    def weights(self):
        return [np.vstack((layer.biases.reshape(1, -1), layer.weights)) for layer in self.model.layers]


class ANNClassification:
    def __init__(self, units, lambda_, lr=0.1, n_epochs=20):
        ...
    def fit(self, X, y):
        ...
    def predict(self, X):
        ...

if __name__ == "__main__":
    data = np.random.randint(-5, 5, (5024, 2))
    y = (data[:, 0] + data[:, 1]) > 1
    y = data.sum(axis=1)
    # data = np.array([[1,2],[2,3],[3,4],[4,5]])
    # y = np.array((0,0,1,1))
    # input = np.array([[1, 2]])
    # target = np.array([[0, 1, 0]])
    # norm = np.linalg.norm(input, axis=1)
    # input_n = input / norm
    # data = input_n
    # y = target

    # net = NN(1, 25, 2, 1, lr=0.01)
    # net.train(data, y, 200)
    # print(net.forward(np.array([[3, 0],[0,1], [-4,4], [4,1]])))
    # print(net.forward(np.array([[2, 0],[1,0],[0,2], [0,4], [0,3],[4,0],[3,0]])))

    # quit()
    np.random.seed(2)
    net = NN(1, [10], 2, 1, lr=1)
    X = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    y = np.array((0, 1, 2, 3))
    hard_y = np.array((0,1,1,0))

    #TODO TODO TODO TODO TODO TODO TODO REGULARIZACIJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    # two hidden layers
    fitter = ANNRegression(units=[13, 6], lambda_=0.0001)
    m = fitter.fit(X, hard_y)
    pred = m.predict(X)
    np.testing.assert_allclose(pred, hard_y, atol=0.01)
    #https: // www.adeveloperdiary.com / data - science / deep - learning / neural - network - with-softmax - in -python /
    https: // medium.com / @ neuralthreads / backpropagation - made - super - easy -
    for -you - part - 2 - 7b2a06f25f3c
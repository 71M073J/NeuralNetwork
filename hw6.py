import random

import numpy as np
import scipy.optimize as op


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SigmoidBack(y, err):
    return err * (y * (1 - y))
    # sig = Sigmoid(y)
    # return err * (sig * (1 - sig))


def cross_E(y_pred, y_true):  # CE
    y_t = np.zeros((y_true.shape[0], y_true.shape[0]))
    y_t[range(len(y_true)), y_true] = 1
    return -np.sum(y_t * np.log(y_pred + 10 ** -100))


def RMSE(ypred, ytrue):
    return np.sqrt(np.power(ytrue - ypred.T, 2).mean())


def cross_E_grad(y_true, y_pred):  # CE derivative
    return -y_true / (y_pred + 10 ** -100)


def ReLUBackward(y, err):
    return err * (y > 0)


def ReLU(x):
    return np.maximum(0, x)


def SoftMaxBack(y, err):
    # I = np.eye(y.shape[0])
    #activation_err = np.zeros(err.shape)
    #for i in range(err.shape[0]):
    #    activation_err[i] = err.dot(np.diagflat(y[i]) - y[i].T.dot(y[i]))
    #    ...
    # return err.dot(SoftMax(y) * (I - SoftMax(y).T)
    # SM = y.reshape((-1, 1))
    # jac = np.diagflat(y) - np.dot(SM, SM.T)
    # err.dot((np.eye(y.shape[0]) - y.T.dot(y)))
    return err.dot(np.diagflat(y) - y.reshape(-1, 1).dot(y.reshape(-1,1).T)).reshape(1,-1)


def SoftMax(x):
    x_exp = np.exp(x - np.max(x, axis=1).reshape(-1,1))
    r = x_exp / np.sum(x_exp, axis=1).reshape(-1,1)
    # r[r == np.nan] = 0
    return r


def Linear(x):
    return x


def LinearBack(y, err):
    return err


class NN:
    def __init__(self, n_hidden_layers, layer_sizes, input_size, output_size, classification=False,
                 batch_size=64, lr=0.01, bfgs=True, lambda_=0.0001):
        #np.random.seed(0)
        self.lambda_ = lambda_
        self.lr = lr
        self.batch_size = batch_size
        self.bfgs = bfgs
        if classification:
            self.loss = cross_E
            self.activation = SoftMax
            self.activation_back = SoftMaxBack
        else:
            self.loss = RMSE
            self.activation_back = ReLUBackward
            self.activation = ReLU

        self.layers = []
        if n_hidden_layers > 0:
            self.layers.append(
                Layer(input_size, layer_sizes[0], self.activation, self.activation_back, self.lr, self.batch_size))
            for i in range(1, len(layer_sizes)):
                self.layers.append(
                    Layer(layer_sizes[i - 1], layer_sizes[i], self.activation, self.activation_back, self.lr,
                          self.batch_size))
                # self.layers.append(Layer(layer_size, output_size, self.activation, self.activation_back, self.lr))
                # self.layers.append(Layer(layer_size, output_size, SoftMax, SoftMaxBack, self.lr, self.batch_size))
            if classification:
                self.layers.append(Layer(layer_sizes[-1], output_size, SoftMax, SoftMaxBack, self.lr, self.batch_size))
            else:
                self.layers.append(Layer(layer_sizes[-1], output_size, Linear, LinearBack, self.lr, self.batch_size))
        else:
            if classification:
                self.layers.append(Layer(input_size, output_size, SoftMax, SoftMaxBack, self.lr, self.batch_size))
            else:
                self.layers.append(Layer(input_size, output_size, Linear, LinearBack, self.lr, self.batch_size))

    def forward(self, input_data):
        temp = input_data.copy()
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                temp = layer.forward(temp, last=True)
            else:
                temp = layer.forward(temp)
        return temp

    def backward(self, err):
        # a - layer input
        # z - a dot weights + biases
        # temp = (true - predicted.T).T
        # temp = err.copy()

        # dzlast = err
        # dWlast = self.layers[-1].previous_activated.T.dot(dzlast)
        # dblast = dzlast
        grads = []
        for layer in reversed(self.layers):
            activation_err = self.activation_back(layer.out, err)  # odvod po ????
            grad = layer.previous_activated.T.dot(activation_err)  # /self.batch_size
            grad = np.vstack((grad, err))
            grads.append(grad)
            err = layer.weights.dot(activation_err.T).T
        grads.reverse()
        gradarr = np.zeros(sum([len(x.reshape(-1)) for x in grads]))
        cur = 0
        for g in grads:
            gradarr[cur:cur + len(g.reshape(-1))] = g.reshape(-1)
            cur += len(g.reshape(-1))
        return gradarr

    def batch_grad(self, X, y):
        prediction = self.forward(np.atleast_2d(X[0, :]))
        predictions = np.zeros((X.shape[0], prediction.shape[0]))
        predictions[0] = prediction
        # supposedly isto za multinomial softmax kot za MSE za regresijo
        gradvec = self.backward(prediction.flatten() - y[0])
        for i in range(1, X.shape[0]):
            prediction = self.forward(np.atleast_2d(X[i, :])).flatten()
            predictions[i] = prediction
            gradvec += self.backward(prediction - y[i])
        return gradvec / X.shape[0], predictions

    def weights_to_loss(self, w, *args):
        X, y = args[0], args[1]
        cur = 0
        for layer in self.layers:
            layer.weights[:, :] = w[cur:cur + layer.n_outputs * layer.n_inputs].reshape(
                (layer.n_inputs, layer.n_outputs))
            cur += layer.n_inputs * layer.n_outputs
            layer.biases[:, :] = w[cur:cur + layer.n_outputs]
            cur += layer.n_outputs
        #prediction = self.forward(X)
        gradients, prediction = self.batch_grad(X, y)
        loss = self.loss(prediction, y)
        return loss, gradients

    def update_weights(self, w):
        cur = 0
        self.updated_weights = w.copy()
        for layer in self.layers:
            layer.weights[:, :] = w[cur:cur + layer.n_outputs * layer.n_inputs].reshape(
                (layer.n_inputs, layer.n_outputs)).copy()
            layer.weights += layer.weights * self.lambda_
            cur += layer.n_inputs * layer.n_outputs
            layer.biases[:, :] = w[cur:cur + layer.n_outputs].copy()
            cur += layer.n_outputs

    def gd(self, x0, X_train, y, i, to, lr):
        loss, g = self.weights_to_loss(x0, X_train, y[i * self.batch_size:to])
        # print(loss)
        self.updated_weights = x0 - g * lr
        cur = 0
        for layer in self.layers:
            layer.weights[:, :] -= (g[cur:cur + layer.n_outputs * layer.n_inputs].reshape(
                (layer.n_inputs, layer.n_outputs)) ) * lr
            cur += layer.n_inputs * layer.n_outputs
            layer.biases[:, :] -= g[cur:cur + layer.n_outputs] * lr
            cur += layer.n_outputs

    def train(self, X, y, n_epochs, test=None, verbose=False):
        scores = []
        mins = 123415123
        minws = 0
        first = True
        for epoch in range(n_epochs):
            if verbose and epoch % 100 == 0: print(f"epoch {epoch}")
            # else: print(epoch, end="")
            for i in range((len(y) // self.batch_size) + 1):
                if i * self.batch_size == len(y): break
                to = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(y) else len(y)
                X_train = X[i * self.batch_size:to, :]
                if first:
                    first = False
                    x0 = np.zeros(
                        sum([(self.layers[i].n_inputs + 1) * self.layers[i].n_outputs for i in range(len(self.layers))]))
                    cur = 0
                    for l in range(len(self.layers)):
                        x0[cur:cur + self.layers[l].n_outputs * self.layers[l].n_inputs] = self.layers[l].weights.flatten()
                        cur += self.layers[l].n_outputs * self.layers[l].n_inputs
                        x0[cur:cur + self.layers[l].n_outputs] = self.layers[l].biases.flatten()
                        cur += self.layers[l].n_outputs
                else:
                    x0 = self.updated_weights.copy()
                # gradient = self.batch_grad(X_train, y[i * self.batch_size:to])
                # ZAKAJ TO NE DELA
                # res = op.fmin_l_bfgs_b(func=self.weights_to_loss, x0=x0, args=(X_train, y[i * self.batch_size:to]),
                #                       approx_grad=False)
                if self.bfgs:
                    #res = op.minimize(fun=self.weights_to_loss, args=(X_train, y[i * self.batch_size:to]), x0=x0, method="Nelder-Mead")
                    res = op.fmin_l_bfgs_b(func=self.weights_to_loss, args=(X_train, y[i * self.batch_size:to]),
                                           x0=x0, factr=10, pgtol=1e-9
                                           )  # ,
                    scores.append(res[1])
                    #print(res[1])
                    #print(self.weights_to_loss(x0, X_train, y[i * self.batch_size:to]))
                    # fprime=self.batch_grad)
                    h = self.weights_to_loss(res[0], X_train, y[i * self.batch_size:to])[0]
                    if h < mins:
                        mins = h
                        self.update_weights(res[0])
                    #print(self.weights_to_loss(self.updated_weights, X_train, y[i * self.batch_size:to])[0])
                    #print(self.forward(X_train).T)

                else:
                    #self.lr *= 10
                    self.gd(x0, X_train, y, i, to, self.lr)
                    #print(self.weights_to_loss(self.updated_weights, X_train, y[i * self.batch_size:to])[0])
                    #print(self.forward(X_train).T)
                # print(res[1])
                # quit()
        #self.update_weights(minws)

class Layer:
    def __init__(self, in_dim, out_dim, activation_fun, activation_back, lr, batch):
        self.weights = np.random.random((in_dim, out_dim))# / (in_dim * out_dim)
        self.n_inputs = in_dim
        self.batch_size = batch
        self.n_outputs = out_dim
        self.biases = np.random.random((1, out_dim)) / (out_dim)
        self.activation = activation_fun
        self.activation_back = activation_back
        self.lr = lr
        self.previous_activated = None
        self.out = None

    def forward(self, input_data, last=False):
        # if last:
        #    self.previous_activated = input_data
        # else:
        self.previous_activated = input_data  # np.hstack((np.ones((input_data.shape[0], 1)), input_data))
        self.out = self.activation(self.previous_activated.dot(self.weights) + self.biases)
        return self.out


class ANNRegression:
    def __init__(self, units, lambda_):
        self.layers = units
        self.lambda_ = lambda_
        self.model = None

    def fit(self, X, y, n_epochs=100, lr=0.1, batch=64):
        self.model = NN(len(self.layers), self.layers, X.shape[1], 1, lr=lr, batch_size=batch)
        self.model.train(X, y, n_epochs)
        return self

    def predict(self, X):
        return self.model.forward(X).flatten()

    def weights(self):
        return [np.vstack((layer.biases.reshape(1, -1), layer.weights)) for layer in self.model.layers]


class ANNClassification:
    def __init__(self, units, lambda_):
        self.layers = units
        self.lambda_ = lambda_
        self.model = None

    def fit(self, X, y, n_epochs=50, lr=0.1, batch=64):
        self.model = NN(len(self.layers), self.layers, X.shape[1], np.bincount(y).shape[0], lr=lr, batch_size=batch,
                        classification=True)
        self.model.train(X, y, n_epochs)
        return self

    def predict(self, X):
        return self.model.forward(X)

    def weights(self):
        return [np.vstack((layer.biases.reshape(1, -1), layer.weights)) for layer in self.model.layers]

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
    # np.random.seed(2)
    net = NN(1, [10], 2, 1, lr=1)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array((0, 1, 2, 3))
    hard_y = np.array((0, 1, 1, 0))

    # TODO TODO TODO TODO TODO TODO TODO REGULARIZACIJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    #fitter = ANNClassification(units=[], lambda_=0.0001)
    #m = fitter.fit(X, y)
    #pred = m.predict(X)
    #np.testing.assert_allclose(pred, np.identity(4), atol=0.01)

    fitter = ANNRegression(units=[], lambda_=0.0001)
    m = fitter.fit(X, y, n_epochs=5, lr=0.1, batch=64)
    pred = m.predict(X)
    print(pred)
    # np.testing.assert_allclose(pred, hard_y, atol=0.01)
    # https: // www.adeveloperdiary.com / data - science / deep - learning / neural - network - with-softmax - in -python /
    # https: // medium.com / @ neuralthreads / backpropagation - made - super - easy -
    # for -you - part - 2 - 7b2a06f25f3c

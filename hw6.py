import random
import time

import numpy as np
import scipy.optimize as op


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SigmoidBack(y, err):
    return err * (y * (1 - y))
    # sig = Sigmoid(y)
    # return err * (sig * (1 - sig))


def cross_E(y_pred, y_true):  # CE
    ce = -np.sum(y_true * np.log(y_pred + 10**-100))
    return ce
    #y_t = np.zeros((y_true.shape[0], y_true.shape[0]))
    #y_t[range(len(y_true)), y_true] = 1
    #return -np.sum(y_t * np.log(y_pred + 10 ** -100))


def RMSE(ypred, ytrue):
    return 0.5 * np.sum(np.power(ytrue - ypred.T, 2)) # why do i need this
    #return np.sqrt(np.power(ytrue - ypred.T, 2).mean())


def cross_E_grad(y_true, y_pred):  # CE derivative
    return -y_true / (y_pred + 10 ** -100)


def ReLUBackward(y, err):
    return err * (y > 0)

def ReLUPrime(y):
    return (y > 0).astype(int)

def ReLU(x):
    return np.maximum(0, x)


def onehot(y):
    y_new = np.zeros((y.shape[0], np.unique(y).shape[0]))
    y_new[range(len(y)), y] = 1
    return y_new.astype(int)

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
        self.lambda_ = lambda_
        self.lr = lr
        self.batch_size = batch_size
        self.bfgs = bfgs
        if classification:
            self.loss = cross_E
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
        #err = prediction - y
        grads = []
        gradz = np.atleast_2d(err).T
        gradw = np.atleast_2d(err).T.dot(self.layers[-1].previous_activated).T
        gradb = err
        dlast = np.vstack((gradw, gradb))
        grads.append(dlast)
        for l, layer in enumerate(reversed(self.layers)):
            if l == 0: continue
            gradz = self.layers[-l].weights.dot(gradz) * ReLUPrime(layer.z).T
            gradw = gradz.dot(layer.previous_activated) + layer.weights.T * self.lambda_ #???
            gradb = gradz
            dlast = np.vstack((gradw.T, gradb.T))
            grads.append(dlast)


        #TO NE DELA VREDU
        #for layer in reversed(self.layers):
        #    activation_err = self.activation_back(layer.out, err)  # odvod po ????
        #    grad = layer.previous_activated.T.dot(activation_err)  # /self.batch_size
        #    grad = np.vstack((grad, err))
        #    grads.append(grad)
        #    err = layer.weights.dot(activation_err.T).T
        grads.reverse()
        gradarr = np.zeros(sum([len(x.reshape(-1)) for x in grads]))
        cur = 0
        for g in grads:
            gradarr[cur:cur + len(g.reshape(-1))] = g.reshape(-1)
            cur += len(g.reshape(-1))
        return gradarr

    def batch_grad(self, X, y):
        prediction = self.forward(np.atleast_2d(X[0, :]))
        predictions = np.zeros((X.shape[0], prediction.shape[1]))
        predictions[0] = prediction
        # supposedly isto za multinomial softmax kot za MSE za regresijo
        gradvec = self.backward(prediction.flatten() - y[0])
        for i in range(1, X.shape[0]):
            prediction = self.forward(np.atleast_2d(X[i, :])).flatten()
            predictions[i] = prediction
            grad_input_i = self.backward(prediction - y[i])
            gradvec += grad_input_i
        return gradvec / X.shape[0], predictions

    def weights_to_loss(self, w, *args):
        X, y = args[0], args[1]
        cur = 0
        saved_weights = []
        saved_biases = []
        for layer in self.layers:
            saved_weights.append(layer.weights.copy())
            saved_biases.append(layer.biases.copy())
            layer.weights[:, :] = w[cur:cur + layer.n_outputs * layer.n_inputs].reshape(
                (layer.n_inputs, layer.n_outputs))
            cur += layer.n_inputs * layer.n_outputs
            layer.biases[:, :] = w[cur:cur + layer.n_outputs]
            cur += layer.n_outputs

        #prediction = self.forward(X)
        gradients, prediction = self.batch_grad(X, y)
        loss = self.loss(prediction, y)
        for i, layer in enumerate(self.layers):
            layer.weights = saved_weights[i].copy()
            layer.biases = saved_biases[i].copy()

        return loss, gradients

    def update_weights(self, w):
        cur = 0
        self.updated_weights = w.copy()
        for layer in self.layers:
            layer.weights[:, :] = w[cur:cur + layer.n_outputs * layer.n_inputs].reshape(
                (layer.n_inputs, layer.n_outputs)).copy()
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

    def check_grad(self, x0, X_train, y, i, to, yt):

        #check gradient
        r = self.weights_to_loss(x0, X_train, y[i * self.batch_size:to]) #returna loss, gradient
        print(r[1])
        test_grad = np.zeros(x0.shape)
        for h in range(x0.shape[0]):
            wi = x0.copy()
            wi[h] += 1e-5
            loss_direction_i = self.weights_to_loss(wi, X_train, yt)[0] / X_train.shape[0]
            grad_dir_i = (loss_direction_i - r[0] / X_train.shape[0])/1e-5
            test_grad[h] += grad_dir_i
        print(test_grad)
        print(abs(test_grad - r[1]))

    def train(self, X, y, n_epochs, test=None, verbose=False, report_epochs=10, tests=True):

        again = False
        mins = 123415123
        minws = 0
        first = True
        for epoch in range(n_epochs):
            # else: print(epoch, end="")
            scores = []
            hs = []
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

                yt = y[i * self.batch_size:to]

                res = op.fmin_l_bfgs_b(func=self.weights_to_loss, args=(X_train, yt),
                                       x0=x0#, factr=1e-5, pgtol=1e-9
                                       )  # ,
                scores.append(res[1])

                self.update_weights(res[0])
                #print(res[1])
                #print(self.weights_to_loss(x0, X_train, y[i * self.batch_size:to]))
                # fprime=self.batch_grad)
                h = self.weights_to_loss(res[0], X_train, y[i * self.batch_size:to])[0]
                if tests:
                    if h + 0.1 < mins:
                        mins = np.mean(hs)
                    else:
                        if again:
                            return
                        else:
                            again = True
                hs.append(h)
            if np.mean(hs) + 0.1 < mins:
                mins = np.mean(hs)
            else:
                if again:
                    return
                else:
                    again = True

            if verbose and epoch % report_epochs == 0: print(f"epoch {epoch}, minloss {mins/X_train.shape[0]}")

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
        self.z = None

    def forward(self, input_data, last=False):
        # if last:
        #    self.previous_activated = input_data
        # else:
        self.previous_activated = input_data  # np.hstack((np.ones((input_data.shape[0], 1)), input_data))
        self.z = self.previous_activated.dot(self.weights) + self.biases
        self.out = self.activation(self.z)
        return self.out


class ANNRegression:
    def __init__(self, units, lambda_):
        self.layers = units
        self.lambda_ = lambda_
        self.model = None

    def fit(self, X, y, n_epochs=50, batch=64, verbose=False, rep_epochs=10, tests=True):
        self.model = NN(len(self.layers), self.layers, X.shape[1], 1, batch_size=batch)
        self.model.train(X, y, n_epochs, verbose=verbose, report_epochs=rep_epochs, tests=tests)
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

    def fit(self, X, y, n_epochs=50, batch=64, verbose=False, rep_epochs=10, tests=True):
        self.model = NN(len(self.layers), self.layers, X.shape[1], np.unique(y).shape[0], batch_size=batch,
                        classification=True)
        y = onehot(y)
        self.model.train(X, y, n_epochs, verbose=verbose, report_epochs=rep_epochs, tests=tests)
        return self

    def predict(self, X):
        return self.model.forward(X)

    def weights(self):
        return [np.vstack((layer.biases.reshape(1, -1), layer.weights)) for layer in self.model.layers]

def load_csvdata(filename):
    if filename == "housing3.csv":
        y_class = True
    else:
        y_class = False
    with open(filename) as f:
        l = f.readline()
        lines = []
        ys = []
        for line in f:
            tokens = line.split(",")
            lines.append([float(x) for x in tokens[:-1]])
            if y_class:
                ys.append(int(tokens[-1][1]) - 1)
            else:
                ys.append(float(tokens[-1]))
    return np.array(lines), np.array(ys)
def three():
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.ensemble import RandomForestClassifier
    m1 = []
    m2 = []
    for i in range(20):
        reg = linear_model.LinearRegression()
        data, cls = load_csvdata("housing2r.csv")
        #print("loaded regression data")
        X_train, X_test, y_train, y_test = train_test_split(data, cls, test_size=0.33, random_state=420)
        nn = ANNRegression([5], 0.01)
        #print("split data, fitting")
        nn.fit(X_train, y_train, verbose=False, rep_epochs=1, batch=999999999, tests=False)
        #print("fit nn")
        reg.fit(X_train, y_train)
        #print("fit both models")
        p1 = nn.predict(X_test)
        p2 = reg.predict(X_test)
        m1.append(np.sqrt(np.mean(np.power(p1 - y_test, 2))))
        m2.append(np.sqrt(np.mean(np.power(p2 - y_test, 2))))
    print("mean, std:")
    print("nn", np.mean(m1), np.std(m1))
    print("linreg", np.mean(m2), np.std(m2))
    m3 = []
    m4 = []
    for i in range(20):
        reg = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=5)
        data, cls = load_csvdata("housing3.csv")
        #print("loaded classification data")
        X_train, X_test, y_train, y_test = train_test_split(data, cls, test_size=0.33, random_state=420)
        nn = ANNClassification([5], 0.001)
        #print("split data, fitting")
        nn.fit(X_train, y_train.astype(int), verbose=False, rep_epochs=1, batch=999999999, tests=False)
        #print("fit nn")
        reg.fit(X_train, y_train)
        #print("fit both models")
        p1 = nn.predict(X_test)
        p2 = reg.predict(X_test)
        m3.append(cross_E(p1, onehot(y_test)))
        m4.append(cross_E(p2, y_test))
    print("mean, std:")
    print("nn", np.mean(m3), np.std(m3))
    print("random forest\n", np.mean(m4), np.std(m4))

def create_final_predictions():
    def load_four_train(name):
        with open(name) as f:
            f.readline()
            lines = []
            ys = []
            for line in f:
                tokens = line.split(",")
                lines.append([float(x) for x in tokens[1:-1]])
                ys.append(int(tokens[-1][-2]) - 1)
        return np.array(lines), np.array(ys)
    def load_four_test(name):
        with open(name) as f:
            f.readline()
            lines = []
            for line in f:
                tokens = line.split(",")
                lines.append([float(x) for x in tokens[1:]])
        return np.array(lines)
    X_train, y_train = load_four_train("train.csv")
    X_test = load_four_test("test.csv")
    print(X_train.shape, y_train.shape)
    nn = ANNClassification([64, 16], 0.1)
    start = time.time()
    nn.fit(X_train, y_train.astype(int), verbose=True, rep_epochs=1, batch=999999999, tests=False)
    end = time.time()
    p = nn.predict(X_test)
    endpred = time.time()
    np.set_printoptions(threshold=np.inf)
    with open("final.txt", "w") as f:
        print(",".join(['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4',
                                         'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']), file=f)
        for i in range(p.shape[0]):
            print(str(i + 1) + "," + ",".join([str(x) for x in p[i]]), file=f)
        print(f"total time elapsed: {end - start} seconds for fitting, \n{endpred - end} seconds for predicting test")


if __name__ == "__main__":
    #three()
    create_final_predictions()
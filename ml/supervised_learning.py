import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from matplotlib import pyplot as plt
import random
from mxnet import autograd


#
# plt.scatter(x_true[:,0],y_data)
# plt.plot(x_true[:,0],x_true[:,0]*4.13+2,'r')
# plt.show()

def data_iter(x, y):
    batch = 10
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch):
        j = nd.array(idx[i:min(i + batch, num_examples)])
        yield nd.take(x, j), nd.take(y, j)


def init():
    w = nd.random_normal(shape=(num_input, 1))
    b = nd.zeros((1))
    return w, b


def net(x, w, b):
    return nd.dot(x, w) + b


def squar_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def real_fun(x):
    return 2 * x[:, 0] - 3.4 * x[:, 1] + 4.2


def plot(losses, x, w, b, samle_size=100):
    xs = list(range(len(losses)))
    f, (fig1, fig2) = plt.subplots(1, 2)
    fig1.set_title('loss curve')
    fig1.plot(xs, losses, '-r')
    fig2.set_title('estimated vs real function')
    fig2.plot(x[:samle_size, 1].asnumpy(), net(x[:samle_size, :], w, b).asnumpy(), 'or', label='estimated')
    fig2.plot(x[:samle_size, 1].asnumpy(), real_fun(x[:samle_size, :]).asnumpy(), '*g', label='real')
    fig2.legend()
    plt.show()


def train(x, w, b, labels, epochs, learning_rate, params):
    niter = 0
    losses = []
    moving_loss = 0
    smoothing_constant = 0.01
    for e in range(epochs):
        total_loss = 0
        for data, label in data_iter(x, labels):
            with autograd.record():
                output = net(data, w, b)
                loss = squar_loss(output, label)
            loss.backward()
            SGD(params, learning_rate)
            total_loss += nd.sum(loss).asscalar()

            niter += 1
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss

            est_loss = moving_loss / (1 - (1 - smoothing_constant) ** niter)
            if (niter + 1) % 100 == 0:
                losses.append(est_loss)
                print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (
                    e, niter, est_loss, total_loss / num_examples))
                plot(losses, x, w, b)


if __name__ == '__main__':
    w_true = [2, -3.4]
    b_true = 4.2
    num_input = 2
    num_examples = 1000
    x = np.random.randn(num_examples, num_input)
    x_true = nd.array(x)
    y_true = x_true[:, 0] * w_true[0] + x_true[:, 1] * w_true[1] + b_true
    y_data = y_true + nd.array(np.random.randn(num_examples))

    w, b = init()
    params = [w, b]
    for param in params:
        param.attach_grad()
    print('start train')
    train(x_true, w, b, y_data, 5, 0.001, params)

    print(w, b)

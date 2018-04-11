# -- coding: utf-8 --

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd
from mxnet import init
import utils1
import random
import matplotlib.pyplot as plt

# function: regularization
# shape is high and samples are small that cause overfitting
# 1. getdata
## y = 0.05 + sum(0.01*xi) + noise
num_train = 20  # samples are small
num_test = 100
num_input = 200  # shape is high


def getData():
    true_w = nd.ones((num_input, 1)) * 0.01
    true_b = 0.05
    x = nd.random_normal(shape=(num_train + num_test, num_input))
    # y = nd.sun([0.01*x[i] for i in range(num_train+num_test)])
    y = nd.dot(x, true_w) + true_b
    y += 0.01 * nd.random_normal(shape=y.shape)
    return x, y


def dataIter(x, y, step):
    idx = list(range(x.shape[0]))
    # print(x.shape[0])
    random.shuffle(idx)
    for j in range(0, x.shape[0], step):
        j = nd.array(idx[j:min(j + step, x.shape[0])])
        # print('j = ',j)
        yield nd.take(x, j), nd.take(y, j)


# 2. model and init
# 2.1 init params(define w,b)
def initParams(num_input):
    w = nd.random_normal(shape=(num_input, 1))
    b = nd.zeros(shape=1)
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params


# 2.2 net = linear layer
def net(x, w, b):
    return nd.dot(x, w) + b


# 3. define loss and optimization
##3.1 loss (add regularization)
# loss = lamda*l2_regular + square_loss
def L2_regular(w, b):
    return (nd.sum(w ** 2) + b ** 2) / 2


def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2


##3.2 optimiz
def sgd(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
    return params


# 4.train
def train(epoches, lr, lamda, x_train, y_train, x_test, y_test, batch_size):
    w, b = params = initParams(num_input)
    train_loss_record = []
    test_loss_record = []
    for e in range(epoches):
        train_loss = 0
        test_loss = 0
        _dataIter = 0
        for data, label in dataIter(x_train, y_train, 1):
            _dataIter += 1

            with autograd.record():
                out = net(data, *params)
                loss = square_loss(out, label) + lamda * L2_regular(*params)
            loss.backward()
            sgd(params, lr / batch_size)
            train_loss += nd.mean(loss).asscalar()
        test_loss = nd.mean(square_loss(net(x_test, *params), y_test)).asscalar()
        train_loss_record.append(train_loss / _dataIter)
        test_loss_record.append(test_loss)
        # print(_dataIter)
        print('epoches: %d, train_loss: %f, test_loss: %f' % (e, train_loss / _dataIter, test_loss))
    return train_loss_record, test_loss_record


if __name__ == '__main__':
    # 1. get data
    x, y = getData()
    x_train, x_test = x[:num_train, :], x[num_train:, :]
    y_train, y_test = y[:num_train], y[num_train:]
    print(x_train.shape, y_train.shape)
    # 1.1 data shuffle
    # for train_data,train_label in dataIter(x_train,y_train,2):
    #     print(train_data.shape)
    # 2. model and init
    w, b = params = initParams(num_input)
    # net = net(x_train[0],*params)
    # print(net)
    epoches = 10
    lr = 0.005
    batch_size = 1
    train_loss, test_loss = train(epoches, lr, 0, x_train, y_train,
                                  x_test, y_test, batch_size)
    train_loss_L, test_loss_L = train(epoches, lr, 5, x_train, y_train,
                                      x_test, y_test, batch_size)
    # plt.plot(train_loss,'b')
    # plt.plot(test_loss,'r')
    # plt.legend(['train','test'])
    # plt.show()
    fig, (fig1, fig2) = plt.subplots(1, 2, sharey=True)
    fig1.plot(train_loss, 'b')
    fig1.plot(test_loss, 'r')
    fig1.legend(['train', 'test'])
    fig2.plot(train_loss_L, 'b')
    fig2.plot(test_loss_L, 'r')
    fig2.legend(['train', 'test'])
    fig2.set_title('l2_regular ')
    fig.show()

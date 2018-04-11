# -- coding: utf-8 --

import mxnet
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn
import utils1
import matplotlib.pyplot as plt
from mxnet import init

# define function: y = 1.2*x -3.4*x**2 + 5.6*x**3
num_train = 100
num_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0


def getData():
    x = nd.random_normal(shape=(num_train + num_test, 1))
    X = nd.concat(x, x ** 2, x ** 3)
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
    # print(y.shape)
    y += 0.1 * nd.random.normal(shape=y.shape)
    return x, X, y


def train(train_x, train_y, test_x, test_y):
    # 1.1 data random and batch
    # print(train_y.shape[0])
    batch = min(10, train_y.shape[0])
    train = gluon.data.ArrayDataset(train_x, train_y)
    data_iter = gluon.data.DataLoader(train, batch_size=batch, shuffle=True)

    # 2.model and init
    net = nn.Sequential()
    net.add(
        nn.Dense(1)  # linear output
    )
    net.initialize(init=init.Xavier())
    # 3. loss and optimization
    l2_loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    epoches = 100
    train_square_loss = []
    test_square_loss = []

    # 4. train and solver
    for e in range(epoches):
        train_loss = 0
        test_loss = 0
        for data, label in data_iter:
            with autograd.record():
                out = net(data)
                loss = l2_loss(out, label)
            loss.backward()
            trainer.step(batch)
            train_loss += nd.mean(loss).asscalar()
        train_square_loss.append(train_loss / len(data_iter))
        test_loss = nd.mean(l2_loss(net(test_x), test_y)).asscalar()
        test_square_loss.append(test_loss)
        print('epoches: %d, train_loss: %f, test_loss: %f' % (
            e, train_loss / len(data_iter), test_loss))
    print('weight: ', net[0].weight.data())
    print('bias: ', net[0].bias.data())
    return train_square_loss, test_square_loss


def mplot(train_loss, test_loss):
    plt.plot(train_loss, 'b')
    plt.plot(test_loss, 'r')
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '__main__':
    # 1.get data
    linear_x, multi_x, y = getData()
    # print(linear_x[:3],multi_x[:3],y[:3])

    # train
    # fit
    train_loss, test_loss = train(multi_x[:num_train, :], y[:num_train],
                                  multi_x[num_train:, :], y[num_train:])
    # mplot(train_loss,test_loss)
    # under_fitting
    train_loss_linear, test_loss_linear = train(linear_x[:num_train, :], y[:num_train],
                                                linear_x[num_train:, :], y[num_train:])
    # mplot(train_loss_linear,test_loss_linear)
    # over_fitting
    train_loss_overfit, test_loss_overfit = train(linear_x[:2, :], y[:2],
                                                  linear_x[num_train:, :], y[num_train:])
    # mplot(train_loss_overfit, test_loss_overfit)
    fig, (fig1, fig2, fig3) = plt.subplots(1, 3, sharey=True)
    fig1.plot(train_loss)
    fig1.plot(test_loss, 'r')
    fig1.legend(['train', 'test'])
    fig2.plot(train_loss_linear)
    fig2.plot(test_loss_linear, 'r')
    fig2.legend(['train', 'test'])
    fig3.plot(train_loss_overfit)
    fig3.plot(test_loss_overfit, 'r')
    fig3.legend(['train', 'test'])
    fig.show()

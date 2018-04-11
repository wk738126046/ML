# -- coding:utf-8 --

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import random
import matplotlib.pyplot as plt

# get data
num_input = 200
num_train = 20
num_test = 100
batch_size = 1


def getDate():
    true_w = nd.random_normal(shape=(num_input, 1)) * 0.01
    true_b = 0.05
    x = nd.zeros(shape=(num_train + num_test, num_input))
    y = nd.dot(x, true_w) + true_b
    y += nd.random_normal(shape=y.shape) * 0.01
    return x, y


def dataIter(x, y, batch):
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    for j in range(0, x.shape[0], batch):
        j = nd.array(idx[j:min(j + batch, x.shape[0])])
        yield nd.take(x, j), nd.take(y, j)


if __name__ == '__main__':
    # 1. get data
    x, y = getDate()
    x_train, x_test = x[:num_train, :], x[num_train:, :]
    y_train, y_test = y[:num_train], y[num_train:]

    # 2. model and init
    net = gluon.nn.Sequential()
    net.add(
        gluon.nn.Dense(1)
    )
    net.initialize()
    # 3. loss (regular) and optimiz
    weight_decay = 5
    l2_loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.005, 'wd': weight_decay})
    # 4.train
    epoches = 10
    train_loss_record = []
    test_loss_record = []
    for e in range(epoches):
        train_loss = 0
        test_loss = 0
        _dataIter = 0
        for data, label in dataIter(x_train, y_train, batch_size):
            _dataIter += 1
            with autograd.record():
                out = net(data)
                loss = l2_loss(out, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
        train_loss_record.append(train_loss / _dataIter)
        test_loss = nd.mean(l2_loss(net(x_test), y_test)).asscalar()
        test_loss_record.append(test_loss)
        print('epoches: %d, train_loss: %f, test_loss: %f'
              % (e, train_loss / _dataIter, test_loss))
    plt.plot(train_loss_record, 'b')
    plt.plot(test_loss_record, 'r')
    plt.legend(['train', 'test'])
    plt.show()

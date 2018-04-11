# -- coding: utf-8 --

import mxnet
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import init
import utils1

num_input = 28 * 28
num_output = 10
batch = 256

if __name__ == '__main__':
    # 1. get data
    train_data, test_data = utils1.load_data_fashion_mnist(batch)

    # 2. model and init
    net = nn.Sequential()
    net.add(
        nn.Flatten(),
        nn.Dense(512, activation='relu'),  # relu/sigmoid/softrelu/tanh
        nn.Dense(10)
    )
    net.initialize(init=init.Xavier())
    # print(net.collect_params())

    # 3. loss function and optimization
    entrory_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .5})

    # 4. train and resolver
    for e in range(5):
        train_loss = 0
        train_acc = 0
        for data, label in train_data:
            with autograd.record():
                out = net(data)
                loss = entrory_loss(out, label)
            loss.backward()
            trainer.step(batch)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils1.accuracy(out, label)
        test_acc = utils1.evaluate_accuracy(test_data, net)
        print('epoch: %d, train_loss: %f, train_acc: %f, test_acc: %f' % (
            e, train_loss / len(train_data), train_acc / len(train_data), test_acc))

    print(net[2].weight.shape, net[2].bias.data())

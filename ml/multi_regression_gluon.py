# -- coding: utf-8 --

import mxnet
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd
import utils1

if __name__ == '__main__':
    batch = 128
    # 1. get data
    train_data, test_data = utils1.load_data_fashion_mnist(batch_size=batch)
    print(len(train_data))
    # 2. define model and initialize
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Flatten(),
            nn.Dense(10)
        )
    net.initialize()

    # 3. define loss
    softmax_loss_entrory = gluon.loss.SoftmaxCrossEntropyLoss()

    # 4. optimization
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    # train
    epoches = 5
    for e in range(epoches):
        train_loss = 0
        train_acc = 0
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_loss_entrory(output, label)
            loss.backward()
            trainer.step(batch)  # lr/batch (because grad are sum of batch)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils1.accuracy(output, label)
        test_acc = utils1.evaluate_accuracy(test_data, net)
        print('Epoch: %d, loss: %f, Train_acc: %f, Test_acc: %f' % (
            e, train_loss / len(train_data), train_acc / len(train_data), test_acc
        ))
    print(net[1].weight.shape)
    print(net[1].bias.data())

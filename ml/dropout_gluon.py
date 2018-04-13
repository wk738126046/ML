# -- coding: utf-8 --
import mxnet
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
import utils1

num_input = 28 * 28
num_output = 10
hidden1 = 256
hidden2 = 256
batch_size = 256


def mynet(is_training=True):
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation='relu'))
    if is_training: net.add(gluon.nn.Dropout(0.2))
    net.add(gluon.nn.Dense(256, activation='relu'))
    if is_training: net.add(gluon.nn.Dropout(0.5))
    net.add(gluon.nn.Dense(10))
    # net.initialize()
    return net


if __name__ == '__main__':

    train_data, test_data = utils1.load_data_fashion_mnist(batch_size)
    entrory_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    lr = 0.5
    net = mynet()
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    epoches = 5
    for e in range(epoches):
        train_loss = 0
        test_acc = 0
        train_acc = 0
        for data, label in train_data:
            with autograd.record():
                out = net(data)
                loss = entrory_loss(out, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils1.accuracy(out, label)
        test_acc = utils1.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            e, train_loss / len(train_data),
            train_acc / len(train_data), test_acc))

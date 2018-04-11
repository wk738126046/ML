# -- coding: utf-8 --

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
import matplotlib.pyplot as plt


# 1. get dataset: fashion mnist 28*28*1
def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


# verbose: false default original data instead that choice batch of data random
# batch : data batch size ,default 256
# size :234*256 pictures
def getDate(verbose=False, batch=256):
    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
    if verbose:
        train_data = gluon.data.DataLoader(mnist_train, batch_size=batch, shuffle=True)
        test_data = gluon.data.DataLoader(mnist_test, batch_size=batch, shuffle=False)
        # print(train_data.__len__())
        return train_data, test_data  # yield batch data (class Dataloader)
    return mnist_train, mnist_test


## show dataset file
def showImage(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


def getTextLables(label):
    textLabels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [textLabels[int(i)] for i in label]  # return all label as a list:[xx,xx,..]


# 2. define model
##2.1 init w,b params and give grad
num_input = 784
num_output = 10
w = nd.random_normal(shape=(num_input, num_output))
b = nd.random_normal(shape=num_output)


def initParam(input, output):
    w = nd.random_normal(shape=(input, output))
    b = nd.random_normal(shape=output)
    return w, b


##2.2 model:(activation) softmax, it can choice max and all others that are normolization
def softmax(x):
    e = nd.exp(x)
    partition = e.sum(axis=1, keepdims=True)  # (row,1),row[0] = sum(x[1,:])
    return e / partition


def net(x):
    return softmax(nd.dot(x.reshape((-1, num_input)), w) + b)


# 3 optimization
##3.1 loss function
def cross_entropy(yhat, y):
    return -nd.pick(nd.log(yhat), y)  # return nd.log(yhat)=0 but index[3] =value (if y[0,0,0,1,0,0])


## compare diffirence of prediction and truth (predic ?= truth)
def accurary(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluateAccuracy(dataIter, net):
    acc = 0
    for data, label in dataIter:
        output = net(data)
        acc += accurary(output, label)
    return acc / len(dataIter)


def sgd(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def train(epochs, lr, train_data, params, batch_size):
    for e in range(epochs):
        train_loss = 0
        train_acc = 0
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = cross_entropy(output, label)
            loss.backward()
            sgd(params, lr / batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accurary(output, label)
        # test acc
        test_acc = evaluateAccuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        e, train_loss / len(train_data), train_acc / len(train_data), test_acc))


def test(data, label):
    output = net(data)
    acc = accurary(output, label)
    return acc, output


if __name__ == '__main__':

    # 1. get dataset
    train_data, test_data = getDate(verbose=True)

    if 0:  # plot 28*28*【：】
        mnist_train, mnist_test = getDate()
        data, label = mnist_train[0:3]
        showImage(data)
        print(getTextLables(label))
    # 2. init w,b
    num_input = 784
    num_output = 10
    w, b = initParam(num_input, num_output)
    print(b)
    params = [w, b]
    ## attach grid
    for param in params:
        param.attach_grad()
    acc = evaluateAccuracy(test_data, net)
    print('init acc: ', acc)
    train(5, 0.1, train_data, params, batch_size=256)
    print('complete!')
    print(w.shape, b)
    mnist_train, mnist_test = getDate()
    data, label = mnist_test[0:9]
    print(label[:])
    num = 0
    for i in range(len(label)):
        acc, output = test(data[i], label[i])
        print('acc:', acc)
        if acc == 1.0:
            num += 1
        print(nd.argmax(output, axis=1))
    print('test acc: ', num / len(label))
    exit(0)

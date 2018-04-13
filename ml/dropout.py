# -- coding: utf-8 --
import mxnet
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import utils1


# keep the value less than prob and new matrix will be extended
def dropout(x, dropProb):
    assert 0 <= dropProb <= 1
    if dropProb == 0:
        return nd.zeros(x.shape)
    mask = nd.random.uniform(0, 1, x.shape) < dropProb
    scale = 1 / dropProb
    return scale * x * mask


num_input = 28 * 28
num_output = 10
hidden1 = 256
hidden2 = 256
weight_scale = 0.01


def initParam(verbose=False):
    w1 = nd.random_normal(shape=(num_input, hidden1), scale=weight_scale)
    b1 = nd.zeros(shape=(hidden1))
    w2 = nd.random_normal(shape=(hidden1, hidden2), scale=weight_scale)
    b2 = nd.zeros(shape=(hidden2))
    w3 = nd.random_normal(shape=(hidden2, num_output), scale=weight_scale)
    b3 = nd.zeros(shape=(num_output))
    params = [w1, b1, w2, b2, w3, b3]
    if verbose:
        for param in params:
            param.attach_grad()
    return params


w1 = nd.random_normal(shape=(num_input, hidden1), scale=weight_scale)
b1 = nd.zeros(shape=(hidden1))
w2 = nd.random_normal(shape=(hidden1, hidden2), scale=weight_scale)
b2 = nd.zeros(shape=(hidden2))
w3 = nd.random_normal(shape=(hidden2, num_output), scale=weight_scale)
b3 = nd.zeros(shape=(num_output))


def net(x, is_training=False):
    # w1, b1, w2, b2, w3, b3 = params = initParam(verbose=True)
    x = x.reshape(shape=(-1, num_input))  # (256,784)
    # print(x.shape)
    x1 = nd.relu(nd.dot(x, w1) + b1)
    if is_training: x1 = dropout(x1, 0.8)
    x2 = nd.relu(nd.dot(x1, w2) + b2)
    if is_training: x2 = dropout(x2, 0.5)
    out = nd.dot(x2, w3) + b3
    return out


if __name__ == '__main__':
    # 1.get data
    batch_size = 256
    train_data, test_data = utils1.load_data_fashion_mnist(batch_size=batch_size)

    # 2. model and init
    # params = initParam(verbose=True)
    params = [w1, b1, w2, b2, w3, b3]
    for param in params:
        param.attach_grad()
    # 3. loss and optimiz(sgd)
    entrory_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    lr = 0.5
    # 4 train
    epoches = 5
    for e in range(epoches):
        train_loss = 0
        test_acc = 0
        train_acc = 0
        for data, label in train_data:
            with autograd.record():
                out = net(data, is_training=True)
                loss = entrory_loss(out, label)
            loss.backward()
            utils1.SGD(params, lr / batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils1.accuracy(out, label)
        test_acc = utils1.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            e, train_loss / len(train_data),
            train_acc / len(train_data), test_acc))

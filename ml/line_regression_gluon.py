# --coding: utf-8 --

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import random

num_examples = 1000
num_input = 2
w_true = [4.21, -8.1]
b_true = 2.2


# def init():
#     w = nd.array(nd.random_normal(shape=(num_input,1)))
#     b = nd.array(nd.zeros(num_input))
#     return w,b

def data(w_true, b_true):
    x_true = nd.random_normal(shape=(num_examples, num_input))
    y_true = x_true[:, 0] * w_true[0] + x_true[:, 1] * w_true[1] + b_true
    print('y_true mean: ', (nd.mean(y_true)))
    y_train = y_true + 0.2 * nd.random_normal(shape=y_true.shape)
    print(nd.mean(y_train))
    return x_true, y_train


# def data_iter(x,y,batch_size):
#     idx = list(range(x.shape))
#     random.shuffle(idx)
#     for i in range(0,num_examples,batch_size):
#         j = nd.array(idx[i:min(i+batch_size,num_examples)])
#         return nd.take(x,j),nd.take(y,j)

# def net(input_data):
#     net = gluon.nn.Sequential()
#     net.add(gluon.nn.Dense(1))
#     return net

# def squre_loss(yhat,y):
#     return (yhat - y.reshape(yhat.shape))**2

# def train(epochs,x,y,batch_size=100):
#     for e in range(epochs):
#         total_loss = 0
#         for data,label in data_iter(x,y,batch_size=batch_size):
#             with autograd.record():
#                 output = net(x)
#                 loss = squre_loss(output,label)
#             gluon.Trainer.step()
#             total_loss += loss
#         print('epoch %d : average loss %f'%(e,total_loss/batch_size))

if __name__ == '__main__':
    # w,b = init()
    train_data, train_label = data(w_true, b_true)
    # net = net(train_data)
    # train(5,train_data,train_label)
    # print(w,b)
    batch_size = 10
    dataset = gluon.data.ArrayDataset(train_data, train_label)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
    # net
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()
    # loss
    loss1 = gluon.loss.L2Loss()
    # trainer
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    # train
    for e in range(10):
        for data, label in data_iter:
            total_loss = 0
            with autograd.record():
                output = net(data)
                loss = loss1(output, label)
            loss.backward()
            # loss.backward()求出的梯度是一个batch的梯度之和，所以需要除以batch_size得到平均梯度，就在trainer.step(bs)
            trainer.step(batch_size)  # Gradient will be normalized by 1/batch_size
            total_loss += nd.sum(loss).asscalar()
        print("Epoch %d, average loss: %f" % (e, total_loss / num_examples))
    print(net[0].weight.data(), net[0].bias.data())

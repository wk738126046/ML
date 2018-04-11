# -- coding:utf-8 --
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
import utils1


# 基于logistic regression增加hidding layer
# 分开实现Softmax和交叉熵损失函数可能导致数值不稳定
#####  参数初始化方法
# 推荐的预处理操作是对数据的每个特征都进行零中心化，然后将其数值范围都归一化到[-1,1]范围之内。
# 使用标准差为\sqrt{2/n}的高斯分布来初始化权重，其中n是输入的神经元数。例如用numpy可以写作：w = np.random.randn(n) * sqrt(2.0/n)。
# 使用L2正则化和随机失活的倒置版本。
# 使用批量归一化。
# https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit

def relu(x):
    return nd.maximum(0, x)


# define net
def net(x):
    x = x.reshape((-1, num_input))
    h1 = relu(nd.dot(x, w1) + b1)
    output = nd.dot(h1, w2) + b2
    return output


if __name__ == '__main__':
    # 1. get data
    batch = 256
    train_data, test_data = utils1.load_data_fashion_mnist(batch)

    # 2. model(def net) and init
    num_input = 28 * 28
    num_output = 10
    num_hidden = 512
    weight_scale = .01
    # 2.1 define w and b
    w1 = nd.random_normal(shape=(num_input, num_hidden), scale=weight_scale)
    b1 = nd.zeros(num_hidden)
    w2 = nd.random_normal(shape=(num_hidden, num_output), scale=weight_scale)
    b2 = nd.zeros(num_output)
    params = [w1, b1, w2, b2]
    # 2.2 attach grad (use backward to auto calc)
    for param in params:
        param.attach_grad()

    # 3. loss function
    ##分开实现Softmax和交叉熵损失函数可能导致数值不稳定,采用gluon库,在softmax后有个减去最大值后归一化过程
    entrory_loss = gluon.loss.SoftmaxCrossEntropyLoss()  # object class

    # train
    learning_rate = 0.03
    for e in range(5):
        train_loss = 0
        train_acc = 0
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = entrory_loss(output, label)
            loss.backward()
            utils1.SGD(params, learning_rate / batch)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils1.accuracy(output, label)
        test_acc = utils1.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            e, train_loss / len(train_data),
            train_acc / len(train_data), test_acc))

    print(w1.shape)
    print(b2)

from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd
from mxnet import init
import math

"""
Focal Loss
"""


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, gama=2.0, eps=1e-5, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self._gama = gama
        self._eps = eps

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.softmax(pred, self._axis)
        if self._sparse_label:
            pred = F.pick(pred, label, axis=self._axis, keepdims=True)
            loss = - ((1 - pred) ** self._gama) * (F.log(pred + self._eps))
        else:

            label = nd.reshape_like(F, label, pred)
            # label = _reshape_like(F, label, pred)
            loss = -F.sum(F.log(pred + self._eps) * label * ((1 - pred) ** self._gama), axis=self._axis, keepdims=True)
        loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


'''
    resnet 164
'''


class Residual_v2_bottleneck(nn.HybridBlock):
    def __init__(self, channels, same_shape=True):
        super(Residual_v2_bottleneck, self).__init__()
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels // 4, kernel_size=1)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels // 4, kernel_size=3, padding=1, strides=strides)
            self.bn3 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels, kernel_size=1)
            # self.bn4 = nn.BatchNorm()

            if not same_shape:
                self.conv4 = nn.Conv2D(channels, 1, strides=strides, use_bias=False)

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.relu(self.bn2(out))
        out = F.relu(self.bn3(self.conv2(out)))
        out = self.conv3(out)
        if not self.same_shape:
            x = self.conv4(x)
        return out + x


class ResNet164_v2(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False):
        super(ResNet164_v2, self).__init__()
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(64, 3, 1, 1))
            # block 2
            for _ in range(27):
                net.add(Residual_v2_bottleneck(64))
            # block 3
            net.add(Residual_v2_bottleneck(128, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(128))
            # block 4
            net.add(Residual_v2_bottleneck(256, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(256))
            # block 5
            net.add(nn.BatchNorm())
            net.add(nn.Activation('relu'))
            net.add(nn.AvgPool2D(pool_size=8),
                    nn.Flatten(),
                    )
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('block %d output: %s' % (i + 1, out.shape))
        return out


'''
    DenseNet
'''


class Bottleneck(nn.HybridBlock):
    def __init__(self, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(
                interChannels,
                kernel_size=1,
                use_bias=False,
                weight_initializer=init.Normal(math.sqrt(2. / interChannels)))
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(
                growthRate,
                kernel_size=3,
                padding=1,
                use_bias=False,
                weight_initializer=init.Normal(
                    math.sqrt(2. / (9 * growthRate))))

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = F.concat(*[x, out], dim=1)
        return out


class SingleLayer(nn.HybridBlock):
    def __init__(self, growthRate):
        super(SingleLayer, self).__init__()
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(
                growthRate,
                kernel_size=3,
                padding=1,
                use_bias=False,
                weight_initializer=init.Normal(
                    math.sqrt(2. / (9 * growthRate))))

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.concat(*[x, out], 1)
        return out


class Transition(nn.HybridBlock):
    def __init__(self, nOutChannels):
        super(Transition, self).__init__()
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(
                nOutChannels,
                kernel_size=1,
                use_bias=False,
                weight_initializer=init.Normal(math.sqrt(2. / nOutChannels)))

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.Pooling(out, kernel=(2, 2), stride=(2, 2), pool_type='avg')
        return out


class DenseNet(nn.HybridBlock):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        with self.name_scope():
            self.conv1 = nn.Conv2D(
                nChannels,
                kernel_size=3,
                padding=1,
                use_bias=False,
                weight_initializer=init.Normal(math.sqrt(2. / nChannels)))
            self.dense1 = self._make_dense(growthRate, nDenseBlocks,
                                           bottleneck)

        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        with self.name_scope():
            self.trans1 = Transition(nOutChannels)

        nChannels = nOutChannels
        with self.name_scope():
            self.dense2 = self._make_dense(growthRate, nDenseBlocks,
                                           bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        with self.name_scope():
            self.trans2 = Transition(nOutChannels)

        nChannels = nOutChannels
        with self.name_scope():
            self.dense3 = self._make_dense(growthRate, nDenseBlocks,
                                           bottleneck)
        nChannels += nDenseBlocks * growthRate

        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.fc = nn.Dense(nClasses)

    def _make_dense(self, growthRate, nDenseBlocks, bottleneck):
        layers = nn.HybridSequential()
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.add(Bottleneck(growthRate))
            else:
                layers.add(SingleLayer(growthRate))
        return layers

    def hybrid_forward(self, F, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.Pooling(
            F.relu(self.bn1(out)),
            global_pool=1,
            pool_type='avg',
            kernel=(8, 8))
        out = self.fc(out)
        return out


'''
data augmentation
'''
from mxnet import image
import numpy as np


def transform_train(data, label):
    im = data.asnumpy()
    im = np.pad(im, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    im = nd.array(im, dtype='float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, rand_mirror=True,
                                    rand_crop=True,
                                    mean=np.array([0.4914, 0.4822, 0.4465]),
                                    std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))  # channel x width x height
    return im, nd.array([label]).astype('float32')


def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                    std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).astype('float32')

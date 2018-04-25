# -- coding:utf-8 --

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import init
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms

import numpy as np
import pandas as pd
import utils1
import sys
import os
import shutil
import numpy as np
import datetime
import matplotlib.pyplot as plt
import netlib


# 1.get data: trainlabel.csv contexts are image_id,label[:],size(n,2)
# vaild_ratio: train data num * valid_ratio = test data belong to train data
##1.1 classify train/valid/test/
def reorganizeCIFAR10Data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    # read train label
    with open(os.path.join(data_dir, label_file), 'r') as f:
        lines = f.readlines()[1:]  # jump first line (it's a label line)
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((int(idx), label) for idx, label in tokens))  # dict keep{'id':label}
    labels = set(idx_label.values())  # length=10,fucntion:extract label and no repeat label

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))  # return all files in train_dir
    num_train_tuning = int(num_train * (1 - valid_ratio))  # train num in train dataset
    assert 0 < num_train_tuning < num_train
    # every label's numbers used to train
    num_train_tuning_per_label = num_train_tuning // len(labels)  # all_data/classes_num
    # print('27 line:',num_train_tuning_per_label)
    # print('labels len:',len(labels))
    label_count = dict()

    def mkdirIfNotExist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # classify train and valid data
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = int(train_file.split('.')[0])
        label = idx_label[idx]
        mkdirIfNotExist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        # choice train data(num_train_turning) and valid data(num_train-turining)
        if label not in label_count or label_count[label] < num_train_tuning_per_label:
            mkdirIfNotExist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1  # classes count
        else:
            mkdirIfNotExist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))

    # classify test data
    mkdirIfNotExist([data_dir, input_dir, 'test', 'unknow'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknow'))


##1.2 enhance image dataset
def enhanceDataFuc():
    transform_train = vision.transforms.Compose([
        # transforms.CenterCrop(32)
        # transforms.RandomFlipTopBottom(),
        # transforms.RandomColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),
        # transforms.RandomLighting(0.0),
        # transforms.Cast('float32'),
        # transforms.Resize(32),
        # random crop scale/ratios
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        # random tranverse on left or right
        transforms.RandomFlipLeftRight(),
        # Converts an image NDArray to a tensor NDArray(0,1) and (H*W*C) changes (C*H*W)
        transforms.ToTensor(),
        # standard
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return transform_train, transform_test


# 2. define model
# 2.1 resnet-18 (crop channels)
class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if self.same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


class MyResnet18(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(MyResnet18, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block1(drop maxpool)
            net.add(
                nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1),
                nn.BatchNorm(),
                nn.Activation(activation='relu')
            )
            # block2
            for _ in range(3):
                net.add(Residual(channels=32))
            # block3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(
                    Residual(channels=64)
                )
            # block4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))
            # block5
            net.add(
                nn.AvgPool2D(pool_size=8),
                nn.Flatten(),
                nn.Dense(num_classes)
            )

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('block %d output: %s' % (i + 1, out.shape))
        return out


def getMyNet(ctx):
    num_outputs = 10
    net = MyResnet18(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net


def getResNet164_v2(ctx, verbose=False):
    num_outputs = 10
    net = netlib.ResNet164_v2(num_outputs, verbose)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net


def getWRN16_8(ctx):
    num_outputs = 10
    net = netlib.WideResnet_16_8(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net

# 4.train
def myTrain(net, batch_size, train_data, valid_data, epoches, lr, wd, ctx, lr_period, lr_decay, verbose=False):
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    prev_time = datetime.datetime.now()
    train_loss_record = []
    valid_loss_record = []  # epoches recycle record loss
    train_acc_record = []
    valid_acc_record = []
    for e in range(epoches):
        train_loss = 0.0
        train_acc = 0.0
        # if e > 99 and e < 251 and e % 10 == 0:
        #     trainer.set_learning_rate(trainer.learning_rate * lr_decay)  # decrease lr
        if e == 60 or e == 120 or e == 160:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)  # decrease lr
        # if e > 150 and e % 20 == 0:
        #     trainer.set_learning_rate(trainer.learning_rate * lr_decay)  # decrease
        # print('train len:',len(train_data))
        train_acc_old = 0
        b = 0
        for data, label in train_data:
            # print('label type:', label.dtype)
            # print('data type:', data)
            # print(len(train_data))
            b += 1
            label = label.reshape(shape=(label.shape[0],))  # be careful:it turns to vector
            label = label.astype('float32').as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entrory(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils1.accuracy(output, label)
            # print('time: %d, acc:%f'%(b,train_acc - train_acc_old))
            # train_acc_old = train_acc

        train_loss_record.append(train_loss / len(train_data))
        train_acc_record.append(train_acc / len(train_data))
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time %02d:%02d:%02d' % (h, m, s)
        if valid_data is not None:
            valid_acc = evaluate_accuracy(valid_data, net, ctx)
            valid_acc_record.append(valid_acc)
            print(len(valid_data))
            if verbose:
                ###valid data loss
                valid_loss = 0
                # print('valid_data len', len(valid_data))
                for data, valid_label in valid_data:
                    valid_label = valid_label.reshape(shape=(valid_label.shape[0],))  # be careful:it turns to vector
                    valid_label = valid_label.astype('float32').as_in_context(ctx)
                    # with autograd.predict_mode():
                    out = net(data.as_in_context(ctx))
                    loss = softmax_cross_entrory(out, valid_label)
                    valid_loss += nd.mean(loss).asscalar()
                    # valid_loss = nd.mean(loss).asscalar( # only used valid loss of every batch(vaild_data)
                valid_loss_record.append(valid_loss / len(valid_data))  # record every batch loss of valid data

                # print('valid loss:', valid_loss/len(valid_test_data))
                epoch_str = ("Epoch %d. Train Loss: %f,Valid Loss: %f, Train acc %f, Valid acc %f, "
                             % (
                             e, train_loss / len(train_data), valid_loss / len(valid_data), train_acc / len(train_data),
                             valid_acc))
            else:
                epoch_str = ("Epoch %d. Train Loss: %f, Train acc %f, Valid acc %f, "
                             % (e, train_loss / len(train_data), train_acc / len(train_data), valid_acc))

        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (e, train_loss / len(train_data), train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + 'lr=' + str(trainer.learning_rate) + ',' + time_str)
    # plot loss and acc
    fig, (fig1, fig2) = plt.subplots(1, 2)
    if verbose:
        fig1.plot(train_loss_record, 'b')
        fig1.legend(['train'])
        fig2.plot(train_acc_record, 'b')
        fig2.legend(['train_acc'])
        if valid_data is not None:
            fig1.plot(valid_loss_record, 'r')
            fig1.legend(['train', 'test'])
            fig2.plot(valid_acc_record, 'r')
            fig2.legend(['train_acc', 'valid_acc'])
    else:
        fig1.plot(train_loss_record, 'b')
        fig1.legend(['train'])
        fig2.plot(train_acc_record, 'b')
        fig2.plot(valid_acc_record, 'r')
        fig2.legend(['train_acc', 'valid_acc'])
    fig.show()
    fig.savefig('./CIFAR10_result.png')


# individual use evaluate acc
def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])


def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    # print(ctx)
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            y = y.astype('float32')  ## change y dtype
            # print('argmax x:',nd.argmax(net(X), axis=1).dtype)
            # acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            acc += nd.sum(nd.argmax(net(X), axis=1) == y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()  # don't push too many operators into backend
    return acc.asscalar() / n


if __name__ == '__main__':
    # print('path',sys.path)
    demo = False
    if demo:
        train_dir = 'train_tiny'
        test_dir = 'test_tiny'
        batch_size = 32
        data_dir = './CIFAR-10-kaggleData'
    else:
        train_dir = 'train'
        test_dir = 'test'
        batch_size = 32
        data_dir = './CIFAR-10-kaggleData/CompleteData_CIFAR10'
    # valid_ratio = 0.1
    # num_epochs = 300
    # learning_rate = 0.1
    # weight_decay = 0.001
    # lr_period = 40
    # lr_decay = 0.5
    label_file = 'trainLabels.csv'
    input_dir = 'train_valid_test'
    # print('train dir:',os.path.join(data_dir,train_dir))
    # print('test dir:', os.path.join(data_dir,test_dir))
    # print('train_test_valid:', os.path.join(data_dir,input_dir,train_dir))
    # print('listdir',len(os.listdir(os.path.join(data_dir, train_dir))))

    # 1.1 classify dataset
    # reorganizeCIFAR10Data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio)

    # 1.2 data augmentation
    input_str = data_dir + '/' + input_dir + '/'
    loader = mx.gluon.data.DataLoader
    UseMyDA = True
    # UseMyDA = False
    if UseMyDA:
        # modify valid data type
        transform_train1, transform_test1 = enhanceDataFuc()
        train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1, transform=netlib.transform_train)
        # valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1,transform=netlib.transform_test)
        valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1)
        train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', flag=1, transform=netlib.transform_train)
        test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, transform=netlib.transform_test)

        train_data = loader(train_ds, batch_size=batch_size, shuffle=True, last_batch='keep')

        valid_data = loader(valid_ds.transform_first(transform_test1), batch_size=batch_size, shuffle=True,
                            last_batch='keep')
        # verify loss of valid data
        train_valid_data = loader(train_valid_ds, batch_size=batch_size, shuffle=True, last_batch='keep')

        test_data = loader(test_ds, batch_size=batch_size, shuffle=False, last_batch='keep')
        print('len valid data:', len(valid_data))

    else:  # use only randomSizeClip
        # print('input dir:',input_str)
        # read original images, flag = 1 mean RGB
        train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1)
        valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1)
        train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', flag=1)
        test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1)

        # data augmentation
        transform_train, transform_test = enhanceDataFuc()

        train_data = loader(train_ds.transform_first(transform_train), batch_size=batch_size, shuffle=True,
                            last_batch='keep')

        valid_data = loader(valid_ds.transform_first(transform_test), batch_size=batch_size, shuffle=True,
                            last_batch='keep')
        print('len valid data:', len(valid_data))
        # verify loss of valid data
        # valid_test_data = loader(valid_ds.transform_first(transform_test),batch_size=batch_size,shuffle=False,last_batch='keep')

        train_valid_data = loader(train_valid_ds.transform_first(transform_train), batch_size=batch_size, shuffle=True,
                                  last_batch='keep')

        test_data = loader(test_ds.transform_first(transform_test), batch_size=batch_size, shuffle=False,
                           last_batch='keep')

    # 2.model and init
    ctx = utils1.try_gpu()
    # net = getMyNet(ctx)
    # net = getResNet164_v2(ctx)
    net = getWRN16_8(ctx)

    valid_ratio = 0.1
    num_epochs = 300
    learning_rate = 0.1
    weight_decay = 0.0005
    lr_period = 40
    lr_decay = 0.2

    net.hybridize()

    # 3. loss and optimiz
    softmax_cross_entrory = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    # trainer = mx.gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr,'momentum':0.9,'wd':wd})
    ## check net
    # x = nd.random_normal(shape=(4, 3, 32, 32), ctx=ctx)
    # out = net(x)
    # print(net.collect_params())
    # print(net)
    preds = []
    num = 0
    if True:
        myTrain(net, batch_size, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period,
                lr_decay, False)
        # myTrain(net,batch_size,train_data,valid_data,num_epochs,learning_rate,weight_decay,ctx,lr_period,lr_decay,True)
        print('--demo--train--end---')
        net.save_params('./CIFAR10_TrainParam.params')
        # net.load_params('./CIFAR10_TrainParam.params',ctx) # read net weights
        print('--save params completed! start detect in kaggle test data--')
        for data, label in test_data:
            num += 1
            if num % 100 == 0:
                print('test data batch detect process:', num)
            output = net(data.as_in_context(ctx))
            preds.extend(nd.argmax(output, axis=1).astype(int).asnumpy())
        sorted_ids = list(range(1, len(test_ds) + 1))
        sorted_ids.sort(key=lambda x: str(x))

        df = pd.DataFrame({'id': sorted_ids, 'label': preds})
        df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
        df.to_csv('./submissions_CIFAR/submission.csv', index=False)

        print('---end---')
        '''
        ## valid data loss
        valid_loss = 0
        print('valid_data len',len(valid_data))
        for valid_data, valid_label in valid_data:
            valid_label = valid_label.astype('float32').as_in_context(ctx)
            with autograd.predict_mode():
                out = net(valid_data.as_in_context(ctx))
                loss = softmax_cross_entrory(out,valid_label)
            # valid_loss += nd.mean(loss).asscalar()
            valid_loss = nd.mean(loss).asscalar()
            # valid_loss_record.append(valid_loss)
            print('valid loss:',valid_loss)
        '''
    else:  # product kaggle file

        myTrain(net, batch_size, train_valid_data, None, num_epochs, learning_rate, weight_decay, ctx, lr_period,
                lr_decay)

        for data, label in test_data:
            num += 1
            if num % 100 == 0:
                print('test data batch detect process:', num)
            output = net(data.as_in_context(ctx))
            preds.extend(nd.argmax(output, axis=1).astype(int).asnumpy())
        sorted_ids = list(range(1, len(test_ds) + 1))
        sorted_ids.sort(key=lambda x: str(x))

        df = pd.DataFrame({'id': sorted_ids, 'label': preds})
        df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
        df.to_csv('./submissions_CIFAR/submission.csv', index=False)

    print('end')

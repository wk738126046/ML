# -- coding: utf-8 --

import pandas as pd
import numpy as np
import mxnet
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
import matplotlib.pyplot as plt

square_loss = gluon.loss.L2Loss()


# use log(out,label) to calc L2loss
def getRmseLog(net, x_train, y_train):
    num_train = x_train.shape[0]
    clipped_pred = nd.clip(net(x_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(nd.log(clipped_pred), nd.log(y_train))).asscalar() / num_train)
    # return nd.sqrt(2*nd.sum(square_loss(clipped_pred,y_train)).asscalar()/num_train)


# def getNet():
#     net = gluon.nn.Sequential()
#     with net.name_scope():
#         net.add(gluon.nn.Dense(1))
#     net.initialize()# use BN
#     return net
#### add hidden 256 and BN(my first net)
# def getNet():
#     net = gluon.nn.Sequential()
#     with net.name_scope():
#         # net.add(gluon.nn.BatchNorm())
#         # net.add(gluon.nn.Dense(128,activation='relu'))
#         net.add(gluon.nn.Dense(128))
#         net.add(gluon.nn.BatchNorm(),
#                 gluon.nn.Activation('relu'))
#         # net.add(gluon.nn.Dropout(0.5))
#         net.add(gluon.nn.Dense(1))
#     net.initialize()  # use BN
#     return net
### second net
def getNet():
    net = gluon.nn.Sequential()
    with net.name_scope():
        # net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Dense(1024, activation='relu'))
        # net.add(gluon.nn.BatchNorm(),
        #         gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dropout(0.6))
        net.add(gluon.nn.Dense(1))
    net.initialize()  # use BN
    return net

def mytrain(net, x_train, y_train, x_test, y_test, epoches, verbose_epoch, lr, weight_decay):
    # net = getNet()
    # net.initialize(init=mxnet.init.Xavier(),force_reinit=True)
    # net.collect_params().initialize(init=mxnet.init.Xavier(),force_reinit=True)
    net.collect_params().initialize(force_reinit=True)
    print('start train')
    train_loss_record = []
    if x_test is not None:
        test_loss_record = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(x_train, y_train)
    dataIter_train = gluon.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': weight_decay})
    train_loss = 0
    kaggle_train_loss = 0
    for e in range(epoches):
        # if e > 30:
        #     lr = lr*0.9
        #     trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': weight_decay})
        for data, label in dataIter_train:
            with autograd.record():
                out = net(data)
                loss = square_loss(out, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()

            kaggle_train_loss = getRmseLog(net, x_train, y_train)
        # print('train_loss: %f'%(train_loss/len(dataIter_train)))
        if e > verbose_epoch:
            print('epoch: %d, kaggle_train_loss: %f' % (e, kaggle_train_loss))
        train_loss_record.append(kaggle_train_loss)
        if x_test is not None:
            cur_test_loss = getRmseLog(net, x_test, y_test)
            test_loss_record.append(cur_test_loss)

    # plt
    plt.plot(train_loss_record, 'b')
    plt.legend(['train'])
    if x_test is not None:
        plt.plot(test_loss_record, 'r')
        plt.legend(['train', 'test'])
    plt.show()
    if x_test is not None:
        return kaggle_train_loss, cur_test_loss
    else:
        return kaggle_train_loss, 0


# K折交叉验证：把train_data 分割成K个样本，K-1个用来训练（k-1次），剩下1个用来测试
# 采用 平均loss来评价model
def kFoldCrossVaild(k, epoches, verbose_epoch, x_train, y_train, lr, weight_decay):
    assert k > 1
    fold_size = x_train.shape[0] // k
    train_loss_sum = 0
    test_loss_sum = 0
    for i in range(k):
        # 顺序选一个fold_size的数据为测试集合，将已经被选中的fold_size序号累加合并作为测试集
        x_val_test = x_train[i * fold_size:(i + 1) * fold_size, :]
        y_val_test = y_train[i * fold_size:(i + 1) * fold_size]

        val_train_defined = False
        for _k in range(k):
            if _k != i:
                x_cur_fold = x_train[_k * fold_size:(_k + 1) * fold_size, :]
                y_cur_fold = y_train[_k * fold_size:(_k + 1) * fold_size]
                if not val_train_defined:
                    x_val_train = x_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:  # 相当于每次把第0次的数据和下一次的数据行合并，最后作为当前数据进行下一次累加
                    x_val_train = nd.concat(x_val_train, x_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = getNet()
        print('train ready')
        train_loss, test_loss = mytrain(net, x_val_train, y_val_train, x_val_test, y_val_test, epoches, verbose_epoch,
                                        lr, weight_decay)
        train_loss_sum += train_loss
        print('k fold Test loss: %f' % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k


# save file of kaggle assess model
def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate, weight_decay):
    net = getNet()
    mytrain(net, X_train, y_train, None, None, epochs,
            verbose_epoch, learning_rate, weight_decay)
    preds = net(x_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('../submissions/submission.csv', index=False)


if __name__ == '__main__':
    # 1. get data
    train = pd.read_csv('../kaggleData/kaggle_house_pred_train.csv')
    test = pd.read_csv('../kaggleData/kaggle_house_pred_test.csv')
    x_all = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                       test.loc[:, 'MSSubClass':'SaleCondition']))

    # 1.1 data preprocess( (x-x.mean())/x.std()) standardization
    num_serials = x_all.dtypes[x_all.dtypes != 'object'].index
    x_all[num_serials] = x_all[num_serials].apply(lambda x: (x - x.mean()) / x.std())

    # one-hot encode(按类别中的数据数量（或条件）展开填充)
    x_all = pd.get_dummies(x_all, dummy_na=True)
    # Nan fill in mean
    x_all = x_all.fillna(x_all.mean())

    # change dtype to ndarry
    num_train = train.shape[0]
    x_train = x_all[:num_train].as_matrix()
    x_test = x_all[num_train:].as_matrix()
    y_train = train.SalePrice.as_matrix()

    x_train = nd.array(x_train)
    y_train = nd.array(y_train).reshape((num_train, 1))
    x_test = nd.array(x_test)

    # 3.define loss and opitimiz
    square_loss = gluon.loss.L2Loss()
    # assess results of kaggle
    k = 5
    epochs = 50
    verbose_epoch = 45
    learning_rate = 0.03
    weight_decay = 170
    #### local test
    if 0:
        train_loss, test_loss = kFoldCrossVaild(k, epochs, verbose_epoch, x_train,
                                                y_train, learning_rate, weight_decay)
        print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %
              (k, train_loss, test_loss))
    ### generate assess file
    else:
        learn(epochs, verbose_epoch, x_train, y_train, test, learning_rate, weight_decay)

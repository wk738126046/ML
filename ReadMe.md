# 1. I used mxnet for my ML
 
 
## 1.1 kaggle first param( house price )

    k = 5
    epochs = 100 
    verbose_epoch = 95
    learning_rate = 0.3
    weight_decay = 3.0

##1.2 net   

     net.add(gluon.nn.Dense(128))
     net.add(gluon.nn.BatchNorm(),
             gluon.nn.Activation('relu'))
     net.add(gluon.nn.Dense(1))

##2.1 secend param

    k = 5
    epochs = 50
    verbose_epoch = 45
    learning_rate = 0.03
    weight_decay = 170

##2.2 net

    net.add(gluon.nn.Dense(1024, activation='relu'))
    net.add(gluon.nn.Dropout(0.5))
    net.add(gluon.nn.Dense(1))
    
##3.1 CIFAR10 param

###3.1.1 firts param( epoch 160 and use lr_decay )
   ```
    num_epochs = 300
    learning_rate = 0.1
    weight_decay = 0.0005
    lr_period = 40
    learning_rate = 0.1
    lr_decay = 0.5
``` 
###3.1.1 secend param
   ```
    num_epochs = 300
    learning_rate = 0.1
    weight_decay = 0.001
    lr_period = 40
    lr_decay = 0.5
    if e > 150 and e % 20 == 0:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)  # decrease lr
    
``` 
Epoch 299. Train Loss: 0.286618, Train acc 0.904632, Valid acc 0.933200, lr=0.00078125,


reference:
https://github.com/SinyerAtlantis/deep_learning_gluon/tree/master/2.%20cnn_cifar10

https://github.com/yinglang/CIFAR10_mxnet

### resnet164_v2 reference
https://github.com/L1aoXingyu/cifar10-gluon
